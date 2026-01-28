# train_omni_embed.py
# Standalone Omni embedding training (shared bi-encoder) with:
# - in-batch contrastive (InfoNCE)
# - DDP global negatives (best-effort; depends on src.loss DistributedContrastiveLoss)
# - freeze vision/audio + drop talker (save thinker-only)
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
import shutil
from functools import partial
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, Trainer, HfArgumentParser
from transformers.trainer_utils import seed_worker
from transformers.utils import logging
import yaml

# Import from VLM2Vec project
from src.arguments import ModelArguments, DataArguments, TrainingArguments as VLMTrainingArguments
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.processor import load_processor, get_backbone_name
from src.model.vlm_backbone.omni_embed import OmniEmbedForConditionalGeneration
from src.model.vlm_backbone.omni_embed.qwen2_5omni_backnone import load_qwen2_5omni_thinker

logger = logging.get_logger(__name__)


# =========================
# DDP helpers
# =========================
def is_ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


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


# =========================
# Trainable stats
# =========================
def log_trainable_stats(model: nn.Module, topk: int = 200):
    total_params = 0
    trainable_params = 0
    trainable_names: List[str] = []
    for name, p in model.named_parameters():
        num = p.numel()
        total_params += num
        if p.requires_grad:
            trainable_params += num
            trainable_names.append(name)

    frozen_params = total_params - trainable_params
    frozen_ratio = (frozen_params / total_params) if total_params > 0 else 0.0

    print("Trainable params stats:")
    print(f"  total_params: {total_params}")
    print(f"  trainable_params: {trainable_params}")
    print(f"  frozen_ratio: {frozen_ratio:.4f}")
    print(f"  trainable_parameter_names (show up to {topk}):")
    for name in sorted(trainable_names)[:topk]:
        print(f"    {name}")
    if len(trainable_names) > topk:
        print(f"    ... ({len(trainable_names) - topk} more)")


# =========================
# Prefix-based freeze / save
# =========================
def set_trainable_for_embedding(
    model: nn.Module,
    train_thinker_model: bool = True,
    train_proj_adapters: bool = False,
):
    """
    You already confirmed real prefixes from safetensors index:
      - thinker.visual
      - thinker.audio_tower
      - thinker.model
      - thinker.lm_head
      - talker.*
    Policy (recommended):
      Freeze: thinker.visual.*, thinker.audio_tower.*, talker.*, thinker.lm_head.*
      Train : thinker.model.* (optionally also train proj/adapter)
    """
    # 1) freeze all by default
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 2) whitelist trainable parts
    if train_thinker_model:
        for name, p in model.named_parameters():
            if "talker." in name:
                continue
            if "thinker.visual." in name or "thinker.audio_tower." in name or "thinker.lm_head." in name:
                continue
            if "thinker.model." in name or "base_model.model." in name or name.startswith("model."):
                p.requires_grad = True

    # optional: if you have projection/adapters you want to tune (rarely needed if you only want LLM)
    if train_proj_adapters:
        proj_keys = ("proj", "project", "projection", "adapter", "lora", "pool")
        for name, p in model.named_parameters():
            if name.startswith("thinker.") and any(k in name.lower() for k in proj_keys):
                p.requires_grad = True

    # 3) hard-freeze heads & talker explicitly (for safety / readability)
    for name, p in model.named_parameters():
        if "talker." in name:
            p.requires_grad = False
        if "thinker.visual." in name:
            p.requires_grad = False
        if "thinker.audio_tower." in name:
            p.requires_grad = False
        if "thinker.lm_head." in name:
            p.requires_grad = False


def build_thinker_state_dict(full_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only thinker.* weights; drop talker.* entirely."""
    return {k: v for k, v in full_state_dict.items() if k.startswith("thinker.")}


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
    Processor is attached later (encoder.processor = processor).
    """
    def __init__(
        self,
        model_name_or_path: str,
        model_type: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        device_map=None,
        pooling: str = "mean",
        normalize: bool = True,
        # training policy
        train_proj_adapters: bool = False,
        # LoRA
        lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[str] = None,
        # full finetune (no freezing)
        full_finetune: bool = False,
    ):
        super().__init__()
        if lora and full_finetune:
            raise ValueError("LoRA and full_finetune cannot both be enabled.")
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg = null_tp_plan(cfg)

        # best-effort: disable talker/audio output in config (does NOT remove parameters by itself)
        if hasattr(cfg, "enable_talker"):
            cfg.enable_talker = False
        if hasattr(cfg, "enable_audio_output"):
            cfg.enable_audio_output = False

        if model_type == "qwen2_5_omni":
            base_model = load_qwen2_5omni_thinker(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            base_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        if lora:
            from peft import LoraConfig, get_peft_model

            target_modules = lora_target_modules.split(",") if lora_target_modules else []
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
            )
            # Inject LoRA only into the text model (thinker.model), not vision/audio towers.
            if hasattr(base_model, "model") and base_model.model is not None:
                base_model.model = get_peft_model(base_model.model, lora_config)
            else:
                base_model = get_peft_model(base_model, lora_config)

        self.model = OmniEmbedForConditionalGeneration(base_model)

        if full_finetune:
            for _, p in self.model.named_parameters():
                p.requires_grad = True
        elif lora:
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for name, p in self.model.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
        else:
            # IMPORTANT: prefix-based freeze based on your confirmed prefixes
            set_trainable_for_embedding(
                self.model,
                train_thinker_model=True,
                train_proj_adapters=train_proj_adapters,
            )

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
        dev = self.device
        # Drop non-model keys from collator
        inputs.pop("text", None)
        inputs.pop("global_dataset_name", None)
        def _stack_with_fill(vals):
            if not isinstance(vals, list):
                return vals
            template = next((v for v in vals if v is not None), None)
            if template is None:
                return None
            if not isinstance(template, torch.Tensor):
                template = torch.as_tensor(template)
            stacked = []
            for v in vals:
                if v is None:
                    stacked.append(torch.zeros_like(template))
                else:
                    stacked.append(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
            return torch.stack(stacked, dim=0)

        for key in ("input_features", "audio_attention_mask", "audio_feature_lengths"):
            if key in inputs:
                inputs[key] = _stack_with_fill(inputs[key])

        if isinstance(inputs.get("input_features"), torch.Tensor) and isinstance(inputs.get("audio_attention_mask"), torch.Tensor):
            feat_t = inputs["input_features"].shape[-1]
            mask_t = inputs["audio_attention_mask"].shape[-1]
            if feat_t != mask_t:
                min_t = min(feat_t, mask_t)
                inputs["input_features"] = inputs["input_features"][..., :min_t]
                inputs["audio_attention_mask"] = inputs["audio_attention_mask"][:, :min_t]
            if inputs["input_features"].shape[-1] == 0:
                inputs.pop("input_features", None)
                inputs.pop("audio_attention_mask", None)
                inputs.pop("audio_feature_lengths", None)
            else:
                audio_lens = inputs["audio_attention_mask"].sum(dim=1)
                if torch.any(audio_lens <= 0):
                    inputs.pop("input_features", None)
                    inputs.pop("audio_attention_mask", None)
                    inputs.pop("audio_feature_lengths", None)

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev)

        if not hasattr(self, "_debug_fwd_once"):
            self._debug_fwd_once = True
            devs = {k: (v.device if isinstance(v, torch.Tensor) else type(v)) for k, v in inputs.items()}
            shapes = {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else None) for k, v in inputs.items()}
            print(f"[DEBUG] fwd devices: {devs}")
            print(f"[DEBUG] fwd shapes: {shapes}")

        # No generation; just hidden states
        try:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        except TypeError:
            outputs = self.model(**inputs)

        hs = self._extract_hidden(outputs)

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
    Loss is selected in Trainer after dist init.
    """
    def __init__(self, encoder: OmniEmbedder, temperature: float = 0.02, loss_fn: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.temperature = float(temperature)
        self.loss_fn = loss_fn

    @property
    def device(self):
        return self.encoder.device

    def encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.encoder(**batch)

    def forward(self, qry: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("OmniBiEncoder.loss_fn is None. It should be set by OmniEmbedTrainer before training.")
        q = self.encode(qry)
        p = self.encode(tgt)
        return self.loss_fn(q, p)


# =========================
# Trainer
# =========================
class OmniEmbedTrainer(Trainer):
    """
    Changes vs vanilla Trainer:
      1) Choose loss AFTER dist init
      2) Override get_train_dataloader to avoid DataLoaderDispatcher wrapping issues
      3) Scale loss by world_size (matches MMEBTrainer pattern)
      4) Save thinker-only weights (drop talker) for embedding use-case
    """
    def __init__(self, *args, max_length=None, model_args=None, save_thinker_only: bool = True, **kwargs):
        self.max_length = max_length
        self.model_args = model_args
        self.save_thinker_only = save_thinker_only
        super().__init__(*args, **kwargs)

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

        temperature = getattr(self.model, "temperature", None)
        if temperature is None and self.model_args is not None:
            temperature = getattr(self.model_args, "temperature", 0.02)
        if temperature is None:
            temperature = 0.02

        loss_cls = DistributedContrastiveLoss if (self.is_ddp and dist.get_world_size() > 1) else SimpleContrastiveLoss
        self.model.loss_fn = loss_cls(temperature=float(temperature))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        qry_batch, tgt_batch = inputs
        loss = model(qry=qry_batch, tgt=tgt_batch)
        if not hasattr(self, "_debug_loss_once"):
            self._debug_loss_once = True
            q_ids = qry_batch.get("input_ids")
            p_ids = tgt_batch.get("input_ids")
            q_size = int(q_ids.shape[0]) if isinstance(q_ids, torch.Tensor) else 0
            p_size = int(p_ids.shape[0]) if isinstance(p_ids, torch.Tensor) else 0
            batch_task = "unknown"
            gd = qry_batch.get("global_dataset_name")
            if isinstance(gd, list) and gd:
                batch_task = gd[0]
            print("num_valid_labels", q_size)
            print("loss_full_precision", f"{loss.item():.12f}")
            print("batch_task", batch_task)
            print(
                "has_video",
                (qry_batch.get("pixel_values_videos") is not None) or (tgt_batch.get("pixel_values_videos") is not None),
                "has_audio",
                (qry_batch.get("input_features") is not None) or (tgt_batch.get("input_features") is not None),
            )
            print("num_pairs", q_size * p_size, "num_pos", min(q_size, p_size))
        if isinstance(loss, torch.Tensor):
            device = next(model.parameters()).device
            if loss.device != device:
                loss = loss.to(device)
        loss = loss / float(self._dist_loss_scale_factor)
        return (loss, None) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Ensure we do NOT drop columns (tuple/dicts batches)
        # Prefer setting training_args.remove_unused_columns=False outside.
        try:
            if getattr(self.args, "remove_unused_columns", None):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
        except Exception:
            pass

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
            dataloader_params["drop_last"] = True
            dataloader_params["prefetch_factor"] = None

        return DataLoader(train_dataset, **dataloader_params)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        enc: OmniEmbedder = self.model.encoder
        model_name_or_path = getattr(self.model_args, "model_name", None) if self.model_args is not None else None
        if self.model_args is not None and getattr(self.model_args, "lora", False):
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError("LoRA is enabled but PEFT is not available.") from e

            peft_model = None
            peft_source = None
            candidates = [
                ("enc.model.base_model", getattr(enc.model, "base_model", None)),
                ("enc.model.model", getattr(enc.model, "model", None)),
            ]
            base_model = getattr(enc.model, "base_model", None)
            if base_model is not None:
                candidates.append(("enc.model.base_model.model", getattr(base_model, "model", None)))
            for name, m in candidates:
                if m is None:
                    continue
                if isinstance(m, PeftModel) or hasattr(m, "peft_config"):
                    peft_model = m
                    peft_source = name
                    break

            if peft_model is None:
                raise RuntimeError("LoRA is enabled but no PEFT adapter was found on enc.model.")

            logger.info(f"Saving LoRA adapter from {peft_source}.")
            peft_model.save_pretrained(output_dir, safe_serialization=True)
            adapter_cfg = os.path.join(output_dir, "adapter_config.json")
            adapter_weights = os.path.join(output_dir, "adapter_model.safetensors")
            if not (os.path.exists(adapter_cfg) and os.path.exists(adapter_weights)):
                raise RuntimeError(
                    "LoRA save failed: adapter_config.json or adapter_model.safetensors is missing. "
                    "LoRA may not be injected or output_dir is incorrect."
                )

            try:
                if hasattr(enc, "processor") and enc.processor is not None:
                    enc.processor.save_pretrained(output_dir)
            except Exception:
                pass

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if self._should_export_full_dir(output_dir):
                self._export_hf_full_dir(
                    output_dir=output_dir,
                    enc=enc,
                    model_name_or_path=model_name_or_path,
                    adapter_dir=output_dir,
                )
            return

        if state_dict is None:
            state_dict = enc.model.state_dict()

        if self.save_thinker_only:
            # drop talker entirely
            state_dict = build_thinker_state_dict(state_dict)
            # Save as a plain checkpoint (safe for any remote code)
            torch.save(state_dict, os.path.join(output_dir, "thinker_only.bin"))
        else:
            # fallback: save whole model
            try:
                enc.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            except Exception:
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        # Save processor
        try:
            if hasattr(enc, "processor") and enc.processor is not None:
                enc.processor.save_pretrained(output_dir)
        except Exception:
            pass

        # Save config.json (useful for reconstruction)
        try:
            if hasattr(enc.model, "config") and enc.model.config is not None:
                enc.model.config.to_json_file(os.path.join(output_dir, "config.json"))
        except Exception:
            pass

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self._should_export_full_dir(output_dir):
            self._export_hf_full_dir(
                output_dir=output_dir,
                enc=enc,
                model_name_or_path=model_name_or_path,
                adapter_dir=None,
            )

    def _should_export_full_dir(self, output_dir: str) -> bool:
        # Export on final save, and optionally once at a specific checkpoint for quick validation.
        if output_dir == self.args.output_dir:
            return True
        export_step = int(getattr(self.args, "export_full_checkpoint", 0) or 0)
        if export_step <= 0:
            return False
        return os.path.basename(output_dir) == f"checkpoint-{export_step}"

    def _export_hf_full_dir(
        self,
        output_dir: str,
        enc: OmniEmbedder,
        model_name_or_path: Optional[str],
        adapter_dir: Optional[str],
    ):
        if not model_name_or_path or not os.path.isdir(model_name_or_path):
            logger.info("Skip HF export: base model directory is missing.")
            return

        def _copy_if_exists(fname: str):
            src = os.path.join(model_name_or_path, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        # Copy base tokenizer/processor/config files if present.
        copy_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
            "chat_template.jinja",
            "generation_config.json",
            "README.md",
            "LICENSE",
            ".gitattributes",
        ]
        for fname in copy_files:
            _copy_if_exists(fname)

        model_backbone = getattr(self.model_args, "model_backbone", None) if self.model_args is not None else None
        model_type = getattr(self.model_args, "model_type", None) if self.model_args is not None else None
        is_qwen2_5_omni = model_backbone == "qwen2_5_omni" or model_type == "qwen2_5_omni"

        if is_qwen2_5_omni:
            from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

            with torch.no_grad():
                full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                )

                if adapter_dir:
                    from peft import LoraConfig, PeftModel

                    lora_config = LoraConfig.from_pretrained(adapter_dir)
                    peft_model = PeftModel.from_pretrained(
                        full_model.thinker.model,
                        adapter_dir,
                        config=lora_config,
                        is_trainable=False,
                    )
                    merged = peft_model.merge_and_unload()
                    full_model.thinker.model = merged
                else:
                    src_thinker = getattr(enc.model, "base_model", None) or enc.model
                    full_model.thinker.load_state_dict(src_thinker.state_dict(), strict=False)

                full_model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        else:
            with torch.no_grad():
                try:
                    enc.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
                except Exception:
                    torch.save(enc.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


# =========================
# Main
# =========================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Required when collator returns tuple/dicts
    try:
        training_args.remove_unused_columns = False
    except Exception:
        pass

    # If you only want thinker.model training, keep this False.
    # If you need small adapters/projections trainable, set True.
    train_proj_adapters = False

    encoder = OmniEmbedder(
        model_name_or_path=model_args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        pooling="mean",
        normalize=True,
        train_proj_adapters=train_proj_adapters,
        lora=getattr(model_args, "lora", False),
        lora_r=getattr(model_args, "lora_r", 16),
        lora_alpha=getattr(model_args, "lora_alpha", 64),
        lora_dropout=getattr(model_args, "lora_dropout", 0.1),
        lora_target_modules=getattr(model_args, "lora_target_modules", None),
        full_finetune=getattr(model_args, "full_finetune", False),
    )

    # VLM2Vec compatibility
    model_backbone = get_backbone_name(encoder.model.config)
    setattr(model_args, "model_backbone", model_backbone)
    setattr(training_args, "model_backbone", model_backbone)

    processor = load_processor(model_args, data_args)
    setattr(encoder, "processor", processor)

    model = OmniBiEncoder(encoder=encoder, temperature=model_args.temperature, loss_fn=None)
    log_trainable_stats(model)

    with open(data_args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)
        if data_args.data_basedir:
            for _, task_config in dataset_config.items():
                image_dir = task_config.get("image_dir")
                if image_dir and not os.path.isabs(image_dir):
                    task_config["image_dir"] = os.path.join(data_args.data_basedir, image_dir)
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)

    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    trainer = OmniEmbedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=getattr(data_args, "max_len", None),
        model_args=model_args,
        save_thinker_only=True,  # always drop talker on save
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
