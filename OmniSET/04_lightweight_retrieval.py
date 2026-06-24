#!/usr/bin/env python3
"""Lightweight retrieval demo from precomputed OmniSET embeddings.

Default behavior:
- Load query embedding from NPZ (first sample of direction t2i by default).
- Retrieve top-10 candidates from target modality.

Optional behavior:
- Provide --query-text and --instruction to encode a custom text query
  (source modality must be text: --source-modality t).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omniset_pipeline.io_utils import l2_normalize

MODALITIES = ("t", "i", "v", "a")
MODALITY_NAME = {
    "t": "text",
    "i": "image",
    "v": "video",
    "a": "audio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight retrieval demo over OmniSET embeddings.")
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step1_embeddings" / "embeddings_and_queries.npz",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "MMEB-V3" / "omniset",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("catalog.jsonl"),
        help="Absolute path or path relative to dataset root.",
    )
    parser.add_argument("--source-modality", type=str, default="t", choices=list(MODALITIES))
    parser.add_argument("--target-modality", type=str, default="i", choices=list(MODALITIES))
    parser.add_argument("--query-index", type=int, default=0, help="Index in source-modality query list.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--candidate-pool",
        type=str,
        default="target_only",
        choices=["target_only", "all"],
        help="Retrieve from target modality only or all modalities.",
    )

    parser.add_argument("--query-text", type=str, default=None, help="Custom text query content.")
    parser.add_argument("--instruction", type=str, default=None, help="Custom instruction text.")

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "omni-embed-nemotron-3b",
    )
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="nvomniembed",
        choices=["nvomniembed", "qwen2_5_omni", "qwen3_vl", "wave"],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--text-batch-size", type=int, default=1)

    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output path for retrieval result JSON.",
    )
    return parser.parse_args()


def _resolve_path(base: Path, maybe_rel: Path) -> Path:
    return maybe_rel if maybe_rel.is_absolute() else base / maybe_rel


def _canonical_coco_name(semantic_id: str, suffix: str) -> str:
    try:
        return f"COCO_val2014_{int(semantic_id):012d}.{suffix}"
    except Exception:
        return f"{semantic_id}.{suffix}"


def load_catalog_map(dataset_root: Path, catalog_file: Path) -> Dict[str, Dict[str, object]]:
    path = _resolve_path(dataset_root, catalog_file)
    if not path.exists():
        return {}

    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("image_id", ""))
            if sid:
                out[sid] = row
    return out


def describe_candidate(
    dataset_root: Path,
    catalog_map: Dict[str, Dict[str, object]],
    semantic_id: str,
    modality: str,
) -> Dict[str, str]:
    cat = catalog_map.get(str(semantic_id), {})

    if modality == "t":
        caps = cat.get("captions", [])
        text = str(caps[0]).strip() if isinstance(caps, list) and caps else ""
        if not text:
            text = f"caption_of_{semantic_id}"
        return {"kind": "text", "value": text}

    if modality == "i":
        rel = str(cat.get("image_filename", f"val2014/{_canonical_coco_name(semantic_id, 'jpg')}"))
        return {"kind": "path", "value": str(_resolve_path(dataset_root, Path(rel)))}

    if modality == "v":
        rel = str(cat.get("video_filename", f"videos/{_canonical_coco_name(semantic_id, 'mp4')}"))
        return {"kind": "path", "value": str(_resolve_path(dataset_root, Path(rel)))}

    rel = str(cat.get("audio_filename", f"audios/{_canonical_coco_name(semantic_id, 'wav')}"))
    return {"kind": "path", "value": str(_resolve_path(dataset_root, Path(rel)))}


def load_npz_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Embedding NPZ not found: {path}")
    blob = np.load(path, allow_pickle=True)
    semantic_ids = np.asarray(blob["semantic_id"]).astype(str)
    modalities = np.asarray(blob["modality"]).astype(str)
    embeddings = np.asarray(blob["embedding"], dtype=np.float32)
    return blob, semantic_ids, modalities, embeddings


def get_precomputed_query_embedding(
    blob,
    semantic_ids: np.ndarray,
    modalities: np.ndarray,
    source_modality: str,
    target_modality: str,
    query_index: int,
) -> Tuple[np.ndarray, str]:
    key = f"query_{source_modality}2{target_modality}"
    if key not in blob:
        raise KeyError(f"Missing precomputed query key: {key}")

    q_mat = np.asarray(blob[key], dtype=np.float32)
    if q_mat.ndim != 2:
        raise ValueError(f"Invalid shape for {key}: {q_mat.shape}")
    if query_index < 0 or query_index >= q_mat.shape[0]:
        raise IndexError(f"query-index out of range for {key}: {query_index} not in [0, {q_mat.shape[0]-1}]")

    src_indices = np.where(modalities == source_modality)[0]
    if query_index >= len(src_indices):
        raise IndexError(
            f"query-index out of source modality rows: {query_index} not in [0, {len(src_indices)-1}]"
        )

    q_semantic_id = str(semantic_ids[src_indices[query_index]])
    return np.asarray(q_mat[query_index], dtype=np.float32), q_semantic_id


def encode_custom_text_query(
    model_path: Path,
    model_backbone: str,
    device: str,
    text_batch_size: int,
    query_text: str,
    instruction: str,
) -> np.ndarray:
    from omniset_pipeline.embedding_builder import choose_device, encode_batch, load_model_bundle

    instruction = instruction.strip()
    query_text = query_text.strip()
    if not instruction:
        raise ValueError("instruction cannot be empty when using --query-text")
    if not query_text:
        raise ValueError("query-text cannot be empty")

    torch_device = choose_device(device)
    model, processor, data_args = load_model_bundle(
        model_path=model_path,
        model_backbone=model_backbone,
        device=torch_device,
    )

    prompt = f"Instruction: {instruction}\nQuery text: {query_text}"
    emb = encode_batch(
        model=model,
        processor=processor,
        model_backbone=model_backbone,
        texts=[prompt],
        images=None,
        videos=None,
        audios=None,
        batch_size=max(1, int(text_batch_size)),
        device=torch_device,
        data_args=data_args,
        audio_sample_rate=16000,
    )
    return np.asarray(emb[0], dtype=np.float32)


def retrieve_topk(
    query_emb: np.ndarray,
    semantic_ids: np.ndarray,
    modalities: np.ndarray,
    embeddings: np.ndarray,
    target_modality: str,
    candidate_pool: str,
    topk: int,
) -> List[Dict[str, object]]:
    q = l2_normalize(query_emb.reshape(1, -1))
    cand = l2_normalize(embeddings)

    if candidate_pool == "target_only":
        candidate_indices = np.where(modalities == target_modality)[0]
    else:
        candidate_indices = np.arange(len(modalities), dtype=np.int64)

    if len(candidate_indices) == 0:
        return []

    scores = (q @ cand[candidate_indices].T).reshape(-1)
    k = max(1, min(int(topk), len(candidate_indices)))
    order = np.argsort(-scores)[:k]

    rows: List[Dict[str, object]] = []
    for rank, pos in enumerate(order.tolist(), start=1):
        idx = int(candidate_indices[pos])
        rows.append(
            {
                "rank": rank,
                "index": idx,
                "semantic_id": str(semantic_ids[idx]),
                "modality": str(modalities[idx]),
                "score": float(scores[pos]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    if args.source_modality == args.target_modality:
        raise ValueError("source-modality and target-modality should be different for cross-modal retrieval")

    blob, semantic_ids, modalities, embeddings = load_npz_bundle(args.embedding_npz)
    catalog_map = load_catalog_map(args.dataset_root, args.catalog_file)

    if args.query_text is None:
        query_emb, query_semantic_id = get_precomputed_query_embedding(
            blob=blob,
            semantic_ids=semantic_ids,
            modalities=modalities,
            source_modality=args.source_modality,
            target_modality=args.target_modality,
            query_index=args.query_index,
        )
        query_info = {
            "mode": "precomputed",
            "source_modality": args.source_modality,
            "target_modality": args.target_modality,
            "query_index": int(args.query_index),
            "semantic_id": query_semantic_id,
            "instruction": f"Retrieve the matching {MODALITY_NAME[args.target_modality]} with the same semantics.",
        }
    else:
        if args.source_modality != "t":
            raise ValueError("Custom query text is only supported for source-modality 't' in this lightweight demo")
        instruction = args.instruction or f"Retrieve the matching {MODALITY_NAME[args.target_modality]} with the same semantics."
        query_emb = encode_custom_text_query(
            model_path=args.model_path,
            model_backbone=args.model_backbone,
            device=args.device,
            text_batch_size=args.text_batch_size,
            query_text=args.query_text,
            instruction=instruction,
        )
        query_info = {
            "mode": "custom_text",
            "source_modality": args.source_modality,
            "target_modality": args.target_modality,
            "query_text": args.query_text,
            "instruction": instruction,
        }

    rows = retrieve_topk(
        query_emb=query_emb,
        semantic_ids=semantic_ids,
        modalities=modalities,
        embeddings=embeddings,
        target_modality=args.target_modality,
        candidate_pool=args.candidate_pool,
        topk=args.topk,
    )

    for r in rows:
        extra = describe_candidate(
            dataset_root=args.dataset_root,
            catalog_map=catalog_map,
            semantic_id=str(r["semantic_id"]),
            modality=str(r["modality"]),
        )
        r["content_kind"] = extra["kind"]
        r["content"] = extra["value"]

    print("=" * 80)
    print("Lightweight Retrieval Demo")
    print("=" * 80)
    print(json.dumps(query_info, ensure_ascii=False, indent=2))
    print("-" * 80)
    for r in rows:
        print(
            f"#{r['rank']:>2} | score={r['score']:.4f} | sid={r['semantic_id']} | "
            f"mod={r['modality']} | {r['content']}"
        )

    payload = {
        "query": query_info,
        "topk": rows,
        "settings": {
            "embedding_npz": str(args.embedding_npz),
            "dataset_root": str(args.dataset_root),
            "catalog_file": str(_resolve_path(args.dataset_root, args.catalog_file)),
            "candidate_pool": args.candidate_pool,
            "topk": int(args.topk),
        },
    }

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("-" * 80)
        print(f"Saved retrieval json: {args.save_json}")


if __name__ == "__main__":
    main()
