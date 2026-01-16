# train_omni_embed.py
# Standalone Omni embedding training (shared bi-encoder) with:
# - in-batch contrastive (InfoNCE)
# - DDP global negatives
# - freeze vision/audio towers (best-effort)
# - multimodal mixed inputs (text/image/video/audio optional)
# - processor-based collator (pad/stack in collate-time)
#
# Run (single GPU):
#   python train_omni_embed.py
#
# Run (DDP):
#   torchrun --nproc_per_node=2 train_omni_embed.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk, load_dataset
import yaml
from transformers import HfArgumentParser

# Import from VLM2Vec project
from src.arguments import ModelArguments, DataArguments, TrainingArguments as VLMTrainingArguments
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.processor import load_processor, get_backbone_name


# =========================
# DDP helpers
# =========================
def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def ddp_world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1

def ddp_rank() -> int:
    return dist.get_rank() if is_ddp() else 0

@torch.no_grad()
def all_gather_2d(x: torch.Tensor) -> torch.Tensor:
    """Gather [B,D] -> [B*W,D] across ranks."""
    if not is_ddp():
        return x
    xs = [torch.empty_like(x) for _ in range(ddp_world_size())]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


# =========================
# Loss (Using VLM2Vec loss functions)
# =========================


# =========================
# Model loading helpers
# =========================
def null_tp_plan(cfg):
    # avoid tp-plan issues in some remote code
    if hasattr(cfg, "base_model_tp_plan"):
        cfg.base_model_tp_plan = None
    if hasattr(cfg, "thinker_config") and cfg.thinker_config is not None:
        tc = cfg.thinker_config
        if hasattr(tc, "base_model_tp_plan"):
            tc.base_model_tp_plan = None
        if hasattr(tc, "text_config") and tc.text_config is not None and hasattr(tc.text_config, "base_model_tp_plan"):
            tc.text_config.base_model_tp_plan = None
    return cfg

def is_nvomniembed_config(cfg) -> bool:
    model_type = getattr(cfg, "model_type", None)
    name = cfg.__class__.__name__
    return (model_type in {"nvomniembed", "omni-embed", "omni_embed"}) or ("NVOmniEmbed" in name)

def load_omni_model(model_name_or_path: str, cfg, torch_dtype, device_map):
    # omni-embed-nemotron-3b 通常 remote code -> AutoModel 可加载
    if is_nvomniembed_config(cfg):
        try:
            return AutoModel.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

    return AutoModel.from_pretrained(
        model_name_or_path,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


# =========================
# Freeze strategy (best-effort)
# =========================
def freeze_omni_towers(model: nn.Module, freeze_vision=True, freeze_audio=True, freeze_llm: bool = False):
    """
    Best-effort freezing by parameter name patterns.
    - freeze_vision: freeze vision tower / vision encoder / vit modules
    - freeze_audio : freeze audio encoder / speech modules
    - freeze_llm   : optionally freeze the main language backbone (heavy)
    """
    vision_keys = ["vision", "visual", "vit", "image", "internvideo", "vision_tower", "vision_encoder"]
    audio_keys  = ["audio", "speech", "asr", "wav", "clap", "audio_encoder"]
    llm_keys    = ["model.layers", "transformer", "decoder", "llm", "language_model"]

    for name, p in model.named_parameters():
        lname = name.lower()
        if freeze_vision and any(k in lname for k in vision_keys):
            p.requires_grad = False
        if freeze_audio and any(k in lname for k in audio_keys):
            p.requires_grad = False
        if freeze_llm and any(k in lname for k in llm_keys):
            p.requires_grad = False

    # Always keep common projection/adapters trainable (if exist)
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in ["proj", "project", "projection", "adapter", "lora", "pool"]):
            p.requires_grad = True


# =========================
# Embedding encoder
# =========================
def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class OmniEmbedder(nn.Module):
    """
    Omni multimodal inputs -> [B,D] embeddings (L2 normalized).
    """
    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pooling: str = "mean",
        normalize: bool = True,
        freeze_vision: bool = True,
        freeze_audio: bool = True,
        freeze_llm: bool = False,
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg = null_tp_plan(cfg)

        # disable talker/audio output if exists
        if hasattr(cfg, "enable_talker"):
            cfg.enable_talker = False
        if hasattr(cfg, "enable_audio_output"):
            cfg.enable_audio_output = False

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = load_omni_model(model_name_or_path, cfg, torch_dtype=torch_dtype, device_map=device_map)

        # freeze towers (best-effort)
        freeze_omni_towers(self.model, freeze_vision=freeze_vision, freeze_audio=freeze_audio, freeze_llm=freeze_llm)

        self.pooling = pooling
        self.normalize = normalize

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _extract_hidden(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise ValueError(f"Unknown output type: {type(outputs)}")

    def forward(self, **inputs) -> torch.Tensor:
        # move tensors
        dev = self.device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev)

        # tolerate remote-code kwargs
        try:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        except TypeError:
            outputs = self.model(**inputs)

        hs = self._extract_hidden(outputs)

        # direct embedding
        if hs.dim() == 2:
            emb = hs
        else:
            attn_mask = inputs.get("attention_mask", None)
            if attn_mask is None:
                attn_mask = torch.ones(hs.shape[:2], device=hs.device, dtype=torch.long)
            emb = mean_pool(hs, attn_mask)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


class OmniBiEncoder(nn.Module):
    """
    Shared bi-encoder: one encoder for qry & tgt.
    """
    def __init__(self, encoder: OmniEmbedder, temperature: float = 0.02):
        super().__init__()
        self.encoder = encoder
        # Use VLM2Vec loss functions
        loss_fn_cls = DistributedContrastiveLoss if is_ddp() else SimpleContrastiveLoss
        self.loss_fn = loss_fn_cls(temperature=temperature)

    @property
    def device(self):
        return self.encoder.device

    def encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.encoder(**batch)

    def forward(self, qry: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
        q = self.encode(qry)
        p = self.encode(tgt)
        return self.loss_fn(q, p)


# =========================
# Processor-based collator (Using VLM2Vec MultimodalDataCollator)
# =========================


# =========================
# Trainer
# =========================
class OmniEmbedTrainer(Trainer):
    def __init__(self, *args, max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        qry_batch, tgt_batch = inputs
        loss = model(qry=qry_batch, tgt=tgt_batch)
        return (loss, None) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save only encoder weights (and processor)
        enc: OmniEmbedder = self.model.encoder
        # If you want full wrapper, save self.model.state_dict() instead.
        if state_dict is None:
            state_dict = enc.model.state_dict()

        # Try save_pretrained if remote-code supports it; else torch.save
        try:
            enc.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        except Exception:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        try:
            enc.processor.save_pretrained(output_dir)
        except Exception:
            pass

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


# =========================
# Dataset loading (Using VLM2Vec mixed dataset)
# =========================


# =========================
# Main
# =========================
def main():
    # Parse arguments similar to VLM2Vec
    parser = HfArgumentParser((ModelArguments, DataArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: VLMTrainingArguments

    # Build encoder + bi-encoder
    encoder = OmniEmbedder(
        model_name_or_path=model_args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pooling="mean",
        normalize=True,
        freeze_vision=True,   # ✅ freeze vision tower
        freeze_audio=True,    # ✅ freeze audio tower
        freeze_llm=False,     # optional: if True, only train projector/adapters
    )

    # Set model backbone for VLM2Vec compatibility
    model_backbone = get_backbone_name(encoder.model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)

    model = OmniBiEncoder(encoder=encoder, temperature=model_args.temperature)

    # Load dataset using VLM2Vec mixed dataset approach
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        if data_args.data_basedir:
            for _, task_config in dataset_config.items():
                image_dir = task_config.get('image_dir')
                if image_dir and not os.path.isabs(image_dir):
                    task_config['image_dir'] = os.path.join(data_args.data_basedir, image_dir)
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)

    # Use VLM2Vec MultimodalDataCollator
    train_collator = MultimodalDataCollator(encoder.processor, model_args, data_args, training_args)

    trainer = OmniEmbedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()