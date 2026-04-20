"""Embedding generation for OmniSET pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .constants import MODALITY_ORDER, build_directions


def choose_device(device_arg: str) -> torch.device:
    """Select torch device from cli option."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path: str) -> Image.Image:
    """Read an RGB image."""
    with Image.open(path) as im:
        return im.convert("RGB")


def load_video_frames(path: str, num_frames: int) -> List[Image.Image]:
    """Decode representative frames from a video."""
    if num_frames <= 0:
        num_frames = 1

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            frame_ids = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int).tolist()
        else:
            frame_ids = list(range(num_frames))

        frames: List[Image.Image] = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()

        if frames:
            return frames
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
        if arr.ndim == 3:
            arr = arr[None, ...]
        total = arr.shape[0]
        idx = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
        return [Image.fromarray(arr[i].astype(np.uint8)).convert("RGB") for i in idx]
    except Exception as exc:
        raise RuntimeError(f"Failed to decode video frames: {path}") from exc


def load_audio_wave(path: str, target_sr: int) -> np.ndarray:
    """Load audio waveform in float32 mono and resample if needed."""
    wav = None
    sr = None

    try:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if int(sr) == int(target_sr):
            return wav.astype(np.float32)
    except Exception:
        wav = None
        sr = None

    try:
        import torchaudio

        if wav is None:
            tw, sr = torchaudio.load(path)
            if tw.shape[0] > 1:
                tw = tw.mean(dim=0, keepdim=True)
        else:
            tw = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)

        if int(sr) != int(target_sr):
            tw = torchaudio.functional.resample(tw, int(sr), int(target_sr))
        return tw.squeeze(0).numpy().astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio waveform: {path}") from exc


def encode_batch(
    model,
    processor,
    model_backbone: str,
    texts: Sequence[str],
    images: Sequence[object] | None,
    videos: Sequence[object] | None,
    audios: Sequence[np.ndarray] | None,
    batch_size: int,
    device: torch.device,
    data_args,
    audio_sample_rate: int,
) -> np.ndarray:
    """Encode a multimodal batch and return float32 embeddings."""
    from src.model.processor import NVOMNIEMBED, WAVE, process_vlm_inputs_fns

    process_fn = process_vlm_inputs_fns.get(model_backbone, process_vlm_inputs_fns[NVOMNIEMBED])
    outputs: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        model_inputs = {
            "text": list(texts[start:end]),
            "_resize_min_pixels": data_args.resize_min_pixels,
            "_resize_max_pixels": data_args.resize_max_pixels,
        }
        if images is not None:
            model_inputs["images"] = list(images[start:end])
        if videos is not None:
            model_inputs["videos"] = list(videos[start:end])
        if audios is not None:
            model_inputs["audios"] = list(audios[start:end])
            model_inputs["audio_sample_rate"] = int(audio_sample_rate)
            if model_backbone == WAVE:
                model_inputs["_keep_input_raw_wav"] = True

        proc = process_fn(model_inputs=model_inputs, processor=processor, max_length=None)
        proc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proc.items()}

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    reps = model.encode_input(proc)
            else:
                reps = model.encode_input(proc)
        outputs.append(reps.detach().float().cpu().numpy())

    return np.vstack(outputs).astype(np.float32, copy=False)


def load_model_bundle(
    model_path: Path,
    model_backbone: str,
    device: torch.device,
) -> Tuple[object, object, object]:
    """Load processor/model configured for embedding extraction."""
    from src.arguments import DataArguments, ModelArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model_args = ModelArguments(
        model_name=str(model_path),
        model_backbone=model_backbone,
        pooling="last",
        normalize=True,
    )
    data_args = DataArguments()
    data_args.resize_min_pixels = 28 * 28 * 4
    data_args.resize_max_pixels = 28 * 28 * 1280

    processor = load_processor(model_args, data_args=data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor).to(device)
    model.eval()
    return model, processor, data_args


def build_embeddings_and_queries(
    tuples: Sequence[Dict[str, str]],
    model,
    processor,
    model_backbone: str,
    data_args,
    device: torch.device,
    text_batch_size: int,
    image_batch_size: int,
    video_batch_size: int,
    audio_batch_size: int,
    audio_sample_rate: int,
    video_num_frames: int,
    directions: Sequence[Tuple[str, str]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute candidate embeddings and instruction-conditioned query embeddings."""
    if directions is None:
        directions = build_directions(MODALITY_ORDER)

    semantic_ids = [x["semantic_id"] for x in tuples]
    texts = [x["text"] for x in tuples]

    images = [load_image(x["image"]) for x in tuples]
    videos = [load_video_frames(x["video"], video_num_frames) for x in tuples]
    audios = [load_audio_wave(x["audio"], audio_sample_rate) for x in tuples]

    cand_t = encode_batch(
        model=model,
        processor=processor,
        model_backbone=model_backbone,
        texts=texts,
        images=None,
        videos=None,
        audios=None,
        batch_size=text_batch_size,
        device=device,
        data_args=data_args,
        audio_sample_rate=audio_sample_rate,
    )
    cand_i = encode_batch(
        model=model,
        processor=processor,
        model_backbone=model_backbone,
        texts=["Represent this image for retrieval."] * len(images),
        images=images,
        videos=None,
        audios=None,
        batch_size=image_batch_size,
        device=device,
        data_args=data_args,
        audio_sample_rate=audio_sample_rate,
    )
    cand_v = encode_batch(
        model=model,
        processor=processor,
        model_backbone=model_backbone,
        texts=["Represent this video for retrieval."] * len(videos),
        images=None,
        videos=videos,
        audios=None,
        batch_size=video_batch_size,
        device=device,
        data_args=data_args,
        audio_sample_rate=audio_sample_rate,
    )
    cand_a = encode_batch(
        model=model,
        processor=processor,
        model_backbone=model_backbone,
        texts=["Represent this audio for retrieval."] * len(audios),
        images=None,
        videos=None,
        audios=audios,
        batch_size=audio_batch_size,
        device=device,
        data_args=data_args,
        audio_sample_rate=audio_sample_rate,
    )

    all_semantic_ids = np.asarray(
        semantic_ids + semantic_ids + semantic_ids + semantic_ids,
        dtype=object,
    )
    all_modalities = np.asarray(
        ["t"] * len(semantic_ids)
        + ["i"] * len(semantic_ids)
        + ["v"] * len(semantic_ids)
        + ["a"] * len(semantic_ids),
        dtype=object,
    )
    all_embeddings = np.vstack([cand_t, cand_i, cand_v, cand_a]).astype(np.float32, copy=False)

    target_name = {"t": "text", "i": "image", "v": "video", "a": "audio"}
    query_map: Dict[str, np.ndarray] = {}

    for src, tgt in directions:
        instruction = f"Retrieve the matching {target_name[tgt]} with the same semantics."
        key = f"{src}2{tgt}"

        if src == "t":
            q_texts = [f"Instruction: {instruction}\\nQuery text: {t}" for t in texts]
            query_map[key] = encode_batch(
                model=model,
                processor=processor,
                model_backbone=model_backbone,
                texts=q_texts,
                images=None,
                videos=None,
                audios=None,
                batch_size=text_batch_size,
                device=device,
                data_args=data_args,
                audio_sample_rate=audio_sample_rate,
            )
        elif src == "i":
            query_map[key] = encode_batch(
                model=model,
                processor=processor,
                model_backbone=model_backbone,
                texts=[f"Instruction: {instruction}"] * len(images),
                images=images,
                videos=None,
                audios=None,
                batch_size=image_batch_size,
                device=device,
                data_args=data_args,
                audio_sample_rate=audio_sample_rate,
            )
        elif src == "v":
            query_map[key] = encode_batch(
                model=model,
                processor=processor,
                model_backbone=model_backbone,
                texts=[f"Instruction: {instruction}"] * len(videos),
                images=None,
                videos=videos,
                audios=None,
                batch_size=video_batch_size,
                device=device,
                data_args=data_args,
                audio_sample_rate=audio_sample_rate,
            )
        elif src == "a":
            query_map[key] = encode_batch(
                model=model,
                processor=processor,
                model_backbone=model_backbone,
                texts=[f"Instruction: {instruction}"] * len(audios),
                images=None,
                videos=None,
                audios=audios,
                batch_size=audio_batch_size,
                device=device,
                data_args=data_args,
                audio_sample_rate=audio_sample_rate,
            )
        else:
            raise ValueError(f"Unsupported source modality: {src}")

        query_map[key] = np.asarray(query_map[key], dtype=np.float32)

    return all_semantic_ids, all_modalities, all_embeddings, query_map
