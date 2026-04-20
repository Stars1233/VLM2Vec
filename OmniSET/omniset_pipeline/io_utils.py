"""I/O and dataset helpers for OmniSET pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize along last axis."""
    arr = np.asarray(arr, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return arr / denom


def _resolve_meta_path(dataset_root: Path, meta_file: Optional[Path]) -> Path:
    if meta_file is not None:
        return meta_file if meta_file.is_absolute() else dataset_root / meta_file

    candidates = [
        dataset_root / "mscoco_cmret_all.jsonl",
        dataset_root / "mscoco_cmret.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing metadata jsonl. Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def _load_coco_caption_map(dataset_root: Path) -> Dict[int, str]:
    caption_path = dataset_root / "annotations" / "annotations" / "captions_val2014.json"
    if not caption_path.exists():
        return {}

    with caption_path.open("r", encoding="utf-8") as f:
        caption_blob = json.load(f)

    caption_map: Dict[int, str] = {}
    for ann in caption_blob.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in caption_map:
            caption_map[image_id] = str(ann["caption"]).strip()
    return caption_map


def _load_catalog_map(dataset_root: Path, catalog_file: Optional[Path]) -> Dict[int, Dict[str, object]]:
    if catalog_file is None:
        catalog_path = dataset_root / "catalog.jsonl"
    else:
        catalog_path = catalog_file if catalog_file.is_absolute() else dataset_root / catalog_file

    if not catalog_path.exists():
        return {}

    out: Dict[int, Dict[str, object]] = {}
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                image_id = int(row["image_id"])
            except Exception:
                continue
            out[image_id] = row
    return out


def _to_dataset_path(dataset_root: Path, path_or_name: str) -> Path:
    path_or_name = str(path_or_name)
    if path_or_name.startswith("/"):
        return Path(path_or_name)
    return dataset_root / path_or_name


def _canonical_coco_name(image_id: int, suffix: str) -> str:
    return f"COCO_val2014_{int(image_id):012d}.{suffix}"


def read_mscoco_omni_tuples(
    dataset_root: Path,
    max_samples: int,
    meta_file: Optional[Path] = None,
    catalog_file: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Load valid text-image-video-audio tuples from mscoco-omni files.

    Supported metadata schemas:
    - mscoco_cmret_all.jsonl: rows contain image_id + file_name
    - mscoco_cmret.jsonl: rows contain image_id + qry_text (media paths resolved via catalog.jsonl)
    """
    meta_path = _resolve_meta_path(dataset_root, meta_file)
    caption_map = _load_coco_caption_map(dataset_root)
    catalog_map = _load_catalog_map(dataset_root, catalog_file)

    tuples: List[Dict[str, str]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])

            # Text priority: qry_text/text/caption -> COCO captions file -> catalog captions
            text = ""
            for key in ("qry_text", "text", "caption"):
                value = str(row.get(key, "")).strip()
                if value:
                    text = value
                    break
            if not text:
                text = caption_map.get(image_id, "")
            if not text:
                cap_list = catalog_map.get(image_id, {}).get("captions", [])
                if isinstance(cap_list, list) and cap_list:
                    text = str(cap_list[0]).strip()
            if not text:
                continue

            # Media path resolution
            image_path: Optional[Path] = None
            video_path: Optional[Path] = None
            audio_path: Optional[Path] = None

            if "file_name" in row:
                file_name = str(row["file_name"]).strip()
                image_path = dataset_root / "val2014" / file_name
                video_path = dataset_root / "videos" / file_name.replace(".jpg", ".mp4")
                audio_path = dataset_root / "audios" / file_name.replace(".jpg", ".wav")
            else:
                cat_row = catalog_map.get(image_id, {})
                img_name = str(cat_row.get("image_filename", f"val2014/{_canonical_coco_name(image_id, 'jpg')}"))
                vid_name = str(cat_row.get("video_filename", f"videos/{_canonical_coco_name(image_id, 'mp4')}"))
                aud_name = str(cat_row.get("audio_filename", f"audios/{_canonical_coco_name(image_id, 'wav')}"))
                image_path = _to_dataset_path(dataset_root, img_name)
                video_path = _to_dataset_path(dataset_root, vid_name)
                audio_path = _to_dataset_path(dataset_root, aud_name)

            if not (image_path.exists() and video_path.exists() and audio_path.exists()):
                continue

            tuples.append(
                {
                    "semantic_id": str(image_id),
                    "text": text,
                    "image": str(image_path),
                    "video": str(video_path),
                    "audio": str(audio_path),
                }
            )
            if max_samples > 0 and len(tuples) >= max_samples:
                break

    if not tuples:
        raise RuntimeError(
            "No valid semantic tuples found (text/image/video/audio all required). "
            f"meta_file={meta_path}"
        )
    return tuples


def save_embedding_bundle(
    output_path: Path,
    semantic_ids: np.ndarray,
    modalities: np.ndarray,
    embeddings: np.ndarray,
    query_map: Dict[str, np.ndarray],
    directions: Sequence[Tuple[str, str]],
) -> None:
    """Save embeddings and direction query embeddings to NPZ."""
    payload = {
        "semantic_id": np.asarray(semantic_ids).astype(str),
        "modality": np.asarray(modalities).astype(str),
        "embedding": np.asarray(embeddings, dtype=np.float32),
    }
    for src, tgt in directions:
        key = f"query_{src}2{tgt}"
        if f"{src}2{tgt}" not in query_map:
            raise KeyError(f"Missing direction embeddings: {src}2{tgt}")
        payload[key] = np.asarray(query_map[f"{src}2{tgt}"], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def load_embedding_bundle(
    input_path: Path,
    directions: Sequence[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load embeddings and direction query embeddings from NPZ."""
    blob = np.load(input_path, allow_pickle=True)
    semantic_ids = np.asarray(blob["semantic_id"]).astype(str)
    modalities = np.asarray(blob["modality"]).astype(str)
    embeddings = np.asarray(blob["embedding"], dtype=np.float32)

    query_map: Dict[str, np.ndarray] = {}
    for src, tgt in directions:
        key = f"query_{src}2{tgt}"
        if key not in blob:
            raise KeyError(f"Missing key in {input_path}: {key}")
        query_map[f"{src}2{tgt}"] = np.asarray(blob[key], dtype=np.float32)

    return semantic_ids, modalities, embeddings, query_map
