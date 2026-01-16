import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoProcessor, AutoModel, AutoModelForCausalLM

try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as omni_modeling
    _HAS_QWEN_OMNI = True
except Exception:
    Qwen2_5OmniForConditionalGeneration = None
    omni_modeling = None
    _HAS_QWEN_OMNI = False


def _null_tp_plan(cfg):
    if hasattr(cfg, "base_model_tp_plan"):
        cfg.base_model_tp_plan = None
    if hasattr(cfg, "thinker_config") and cfg.thinker_config is not None:
        tc = cfg.thinker_config
        if hasattr(tc, "base_model_tp_plan"):
            tc.base_model_tp_plan = None
        if hasattr(tc, "text_config") and tc.text_config is not None and hasattr(tc.text_config, "base_model_tp_plan"):
            tc.text_config.base_model_tp_plan = None
    return cfg


def _patch_skip_speakers():
    if not _HAS_QWEN_OMNI:
        return
    Qwen2_5OmniForConditionalGeneration.load_speakers = lambda self, path: None
    if omni_modeling is not None and hasattr(omni_modeling, "check_torch_load_is_safe"):
        omni_modeling.check_torch_load_is_safe = lambda: None


def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _is_nvomniembed_config(cfg) -> bool:
    cfg_name = cfg.__class__.__name__
    model_type = getattr(cfg, "model_type", None)
    if model_type in {"nvomniembed", "omni-embed", "omni_embed"}:
        return True
    if "NVOmniEmbed" in cfg_name:
        return True
    return False


def _load_omni_like_model(model_name_or_path: str, cfg, torch_dtype, device_map):
    if _is_nvomniembed_config(cfg):
        try:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        return model

    if _HAS_QWEN_OMNI:
        try:
            return Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except Exception:
            return AutoModel.from_pretrained(
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


class Qwen25OmniEmbedder(torch.nn.Module):
    """
    Embedding wrapper:
    - If model returns hidden states: mean_pool(last_hidden_state, attention_mask) + L2
    - If model directly returns [B,D] embeddings: L2 directly
    - ✅ expose rep_dim for eval.py pre-allocation
    """

    def __init__(self, model_name_or_path: str, torch_dtype=torch.bfloat16, device_map="auto"):
        super().__init__()

        _patch_skip_speakers()

        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg = _null_tp_plan(cfg)

        if hasattr(cfg, "enable_talker"):
            cfg.enable_talker = False
        if hasattr(cfg, "enable_audio_output"):
            cfg.enable_audio_output = False

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = _load_omni_like_model(
            model_name_or_path,
            cfg=cfg,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # ✅ 让 wrapper 的 config 尽量指向“真正产生 hidden_states 的那份 config”
        # （很多 repo 的顶层 cfg 没 hidden_size，但 model.config / model.model.config 有）
        self.config = getattr(self.model, "config", cfg)

        self._maybe_patch_rotary_dtype()
        self._force_left_padding(self.model)
        self._force_left_padding(getattr(self.model, "model", None))

        # ✅ 关键：暴露 rep_dim，避免 eval.py 分配 [B,0]
        self.rep_dim = int(self._infer_rep_dim())
        if self.rep_dim <= 0:
            raise ValueError(
                f"[Qwen25OmniEmbedder] rep_dim inference failed (got {self.rep_dim}). "
                f"Please inspect config: {type(self.config)}"
            )

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ----------------------------
    # rep_dim inference
    # ----------------------------
    def _infer_rep_dim(self) -> int:
        """
        Try best-effort to find embedding dim without running a forward.
        Covers:
        - config.hidden_size / config.d_model
        - config.text_config.hidden_size
        - config.thinker_config.text_config.hidden_size
        - model.model.config.hidden_size
        """
        def _get(obj, path: str):
            cur = obj
            for p in path.split("."):
                if cur is None:
                    return None
                cur = getattr(cur, p, None)
            return cur

        candidates = [
            _get(self, "config.hidden_size"),
            _get(self, "config.d_model"),
            _get(self, "config.text_config.hidden_size"),
            _get(self, "config.thinker_config.text_config.hidden_size"),
            _get(self, "model.config.hidden_size"),
            _get(self, "model.model.config.hidden_size"),
            _get(self, "model.config.d_model"),
            _get(self, "model.model.config.d_model"),
        ]
        for v in candidates:
            if isinstance(v, int) and v > 0:
                return v
        # 有些 remote-code 会用 embed_dim 命名
        for v in [
            _get(self, "config.embed_dim"),
            _get(self, "model.config.embed_dim"),
            _get(self, "model.model.config.embed_dim"),
        ]:
            if isinstance(v, int) and v > 0:
                return v

        return 0

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _force_left_padding(m):
        if m is None:
            return
        if hasattr(m, "padding_side"):
            m.padding_side = "left"
        if hasattr(m, "config") and hasattr(m.config, "padding_side"):
            m.config.padding_side = "left"

    @staticmethod
    def _has_nonempty(container: dict, key: str) -> bool:
        if key not in container or container[key] is None:
            return False
        v = container[key]
        if isinstance(v, list):
            return not all(item is None for item in v)
        return True

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    @staticmethod
    def _pad_and_stack_3d(feats_list, pad_value=0.0):
        normed = []
        Tmax = 0
        for x in feats_list:
            if x is None:
                continue
            x = Qwen25OmniEmbedder._to_tensor(x)
            if x.dim() == 3 and x.size(0) == 1:
                x = x.squeeze(0)
            if x.dim() != 2:
                raise ValueError(f"input_features item must be [128,T] or [1,128,T], got {tuple(x.shape)}")
            Tmax = max(Tmax, x.size(-1))
            normed.append(x)
        if len(normed) == 0:
            return None
        out = []
        for x in normed:
            pad_t = Tmax - x.size(-1)
            if pad_t > 0:
                x = F.pad(x, (0, pad_t), value=pad_value)
            out.append(x)
        return torch.stack(out, dim=0)

    @staticmethod
    def _pad_and_stack_2d(mask_list, pad_value=0):
        normed = []
        Tmax = 0
        for x in mask_list:
            if x is None:
                continue
            x = Qwen25OmniEmbedder._to_tensor(x)
            if x.dim() == 2 and x.size(0) == 1:
                x = x.squeeze(0)
            if x.dim() != 1:
                raise ValueError(f"feature_attention_mask item must be [T] or [1,T], got {tuple(x.shape)}")
            Tmax = max(Tmax, x.numel())
            normed.append(x)
        if len(normed) == 0:
            return None
        out = []
        for x in normed:
            pad_t = Tmax - x.numel()
            if pad_t > 0:
                x = F.pad(x, (0, pad_t), value=pad_value)
            out.append(x)
        return torch.stack(out, dim=0).long()

    @staticmethod
    def _pad_and_stack_firstdim(values, pad_value=0.0):
        if not isinstance(values, list):
            return values
        if all(v is None for v in values):
            return None

        template = next(v for v in values if v is not None)
        template = Qwen25OmniEmbedder._to_tensor(template)
        tail_shape = template.shape[1:] if template.dim() > 1 else ()
        max_len = template.shape[0] if template.dim() > 0 else 1

        tensors = []
        for v in values:
            if v is None:
                tensors.append(None)
                continue
            t = Qwen25OmniEmbedder._to_tensor(v)
            if t.dim() == 0:
                t = t.view(1)
            if t.dim() > 1 and tail_shape and t.shape[1:] != tail_shape:
                raise ValueError(f"Inconsistent tail shapes: {t.shape[1:]} vs {tail_shape}")
            if not tail_shape and t.dim() > 1:
                tail_shape = t.shape[1:]
            max_len = max(max_len, t.shape[0])
            tensors.append(t)

        def _pad_to(t: torch.Tensor) -> torch.Tensor:
            cur = t.shape[0]
            if cur == max_len:
                return t
            pad_shape = (max_len - cur,) + tuple(t.shape[1:])
            pad = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        stacked = torch.stack(
            [
                torch.zeros((max_len,) + tail_shape, dtype=template.dtype, device=template.device)
                if t is None
                else _pad_to(t)
                for t in tensors
            ],
            dim=0,
        )
        return stacked

    @staticmethod
    def _stack_grid_thw(values):
        if not isinstance(values, list):
            return values
        if all(v is None for v in values):
            return None
        ts = []
        for v in values:
            if v is None:
                ts.append(None)
                continue
            t = Qwen25OmniEmbedder._to_tensor(v).reshape(-1)
            if t.numel() != 3:
                raise ValueError(f"Expected grid_thw numel=3, got shape {tuple(t.shape)} numel={t.numel()}")
            ts.append(t)
        template = next(t for t in ts if t is not None)
        return torch.stack([torch.zeros_like(template) if t is None else t for t in ts], dim=0)

    def _maybe_patch_rotary_dtype(self):
        try:
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni
            from flash_attn.layers.rotary import apply_rotary_emb
        except Exception:
            return

        cls = getattr(qwen_omni, "Qwen2_5OmniVisionFlashAttention2", None)
        if cls is None:
            return

        if getattr(cls._apply_rotary_pos_emb_flashatt, "_m2_patch_done", False):
            return

        def _apply_rotary_pos_emb_flashatt(self, tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            tensor_ = tensor.to(freqs.dtype)
            cos = freqs.cos()
            sin = freqs.sin()
            return apply_rotary_emb(tensor_, cos, sin).type_as(tensor)

        _apply_rotary_pos_emb_flashatt._m2_patch_done = True
        cls._apply_rotary_pos_emb_flashatt = _apply_rotary_pos_emb_flashatt

    def _safe_forward(self, model, **model_input):
        try:
            return model(
                **model_input,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            return model(**model_input)
        except Exception:
            return model(**model_input)

    def _extract_embedding(self, outputs, attn_mask: torch.Tensor):
        hs = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hs = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hs = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            hs = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            hs = outputs
        else:
            raise ValueError(f"Unknown model output type: {type(outputs)}")

        if hs.dim() == 2:
            return F.normalize(hs, p=2, dim=-1)

        if attn_mask is None:
            attn_mask = torch.ones(hs.shape[:2], device=hs.device, dtype=torch.long)
        emb = mean_pool(hs, attn_mask)
        return F.normalize(emb, p=2, dim=-1)

    def _text_only_forward(self, text_only_input: dict):
        self._force_left_padding(self.model)
        self._force_left_padding(getattr(self.model, "model", None))

        inner = getattr(self.model, "model", None)
        if inner is not None and callable(inner):
            return self._safe_forward(inner, **text_only_input)
        return self._safe_forward(self.model, **text_only_input)

    def encode_from_inputs(self, inputs: dict) -> torch.Tensor:
        raw_input = dict(inputs)

        EXTRA_KEYS = {"texts", "images", "audios"}
        model_input = {k: v for k, v in raw_input.items() if k not in EXTRA_KEYS}

        dev = self.device

        def _to_long_tensor(x, device=None):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.long()
                return x.to(device) if device is not None else x
            t = torch.tensor(x, dtype=torch.long)
            return t.to(device) if device is not None else t

        for k in ("input_ids", "attention_mask"):
            if k in model_input:
                model_input[k] = _to_long_tensor(model_input[k], device=dev)

        if "audio_attention_mask" in model_input and "feature_attention_mask" not in model_input:
            model_input["feature_attention_mask"] = model_input.pop("audio_attention_mask")
        if "audio_values" in model_input and "input_features" not in model_input:
            model_input["input_features"] = model_input.pop("audio_values")
        for k in ("audio_values", "audio_features", "audios"):
            model_input.pop(k, None)

        if "input_features" in model_input and isinstance(model_input["input_features"], list):
            model_input["input_features"] = self._pad_and_stack_3d(model_input["input_features"], pad_value=0.0)
        if "feature_attention_mask" in model_input and isinstance(model_input["feature_attention_mask"], list):
            model_input["feature_attention_mask"] = self._pad_and_stack_2d(model_input["feature_attention_mask"], pad_value=0)
        if "audio_feature_lengths" in model_input and isinstance(model_input["audio_feature_lengths"], list):
            model_input["audio_feature_lengths"] = torch.tensor(model_input["audio_feature_lengths"], dtype=torch.long)

        if "input_features" in model_input and isinstance(model_input["input_features"], torch.Tensor):
            model_input["input_features"] = model_input["input_features"].to(device=dev, dtype=torch.float32)
        if "feature_attention_mask" in model_input and isinstance(model_input["feature_attention_mask"], torch.Tensor):
            model_input["feature_attention_mask"] = model_input["feature_attention_mask"].to(device=dev, dtype=torch.long)
        if "audio_feature_lengths" in model_input and isinstance(model_input["audio_feature_lengths"], torch.Tensor):
            model_input["audio_feature_lengths"] = model_input["audio_feature_lengths"].to(device=dev, dtype=torch.long)

        if "pixel_values" in model_input and isinstance(model_input["pixel_values"], list):
            model_input["pixel_values"] = self._pad_and_stack_firstdim(model_input["pixel_values"], pad_value=0.0)
        if "pixel_values_videos" in model_input and isinstance(model_input["pixel_values_videos"], list):
            model_input["pixel_values_videos"] = self._pad_and_stack_firstdim(model_input["pixel_values_videos"], pad_value=0.0)

        if "image_grid_thw" in model_input and isinstance(model_input["image_grid_thw"], list):
            model_input["image_grid_thw"] = self._stack_grid_thw(model_input["image_grid_thw"])
        if "video_grid_thw" in model_input and isinstance(model_input["video_grid_thw"], list):
            model_input["video_grid_thw"] = self._stack_grid_thw(model_input["video_grid_thw"])

        for k, v in list(model_input.items()):
            if isinstance(v, torch.Tensor):
                model_input[k] = v.to(dev)

        has_image = self._has_nonempty(model_input, "pixel_values") or self._has_nonempty(model_input, "image_grid_thw")
        has_video = self._has_nonempty(model_input, "pixel_values_videos") or self._has_nonempty(model_input, "video_grid_thw")
        has_audio = (
            self._has_nonempty(model_input, "input_features")
            or self._has_nonempty(model_input, "feature_attention_mask")
            or self._has_nonempty(model_input, "audio_feature_lengths")
        )
        has_multimodal = has_image or has_video or has_audio

        if has_multimodal:
            outputs = self._safe_forward(self.model, **model_input)
            attn_mask = model_input.get("attention_mask", None)
        else:
            visual_keys = {
                "pixel_values", "image_grid_thw",
                "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
                "input_features", "feature_attention_mask", "audio_feature_lengths",
                "audio_values", "audio_attention_mask", "audio_features", "audios",
            }
            text_only_input = {k: v for k, v in model_input.items() if k not in visual_keys}
            outputs = self._text_only_forward(text_only_input)
            attn_mask = text_only_input.get("attention_mask", None)

        if attn_mask is None:
            attn_mask = None
        elif not isinstance(attn_mask, torch.Tensor):
            attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=dev)
        else:
            attn_mask = attn_mask.to(dev).long()

        return self._extract_embedding(outputs, attn_mask)

    def forward(self, **inputs):
        return self.encode_from_inputs(inputs)