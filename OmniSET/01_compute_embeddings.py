#!/usr/bin/env python3
"""Step 1: compute OmniSET embeddings and save them locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omniset_pipeline.constants import MODALITY_ORDER, build_directions  # noqa: E402
from omniset_pipeline.embedding_builder import (  # noqa: E402
    build_embeddings_and_queries,
    choose_device,
    load_model_bundle,
)
from omniset_pipeline.io_utils import read_mscoco_omni_tuples, save_embedding_bundle  # noqa: E402


MODEL_BACKBONE_CHOICES = ["nvomniembed", "qwen2_5_omni", "qwen3_vl", "wave"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and export OmniSET embeddings bundle.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "datasets" / "MMEB-V3" / "mscoco-omni",
    )
    parser.add_argument(
        "--meta-file",
        type=Path,
        default=Path("mscoco_cmret_all.jsonl"),
        help="Metadata jsonl path (absolute or relative to dataset root).",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("catalog.jsonl"),
        help="Catalog jsonl path (absolute or relative to dataset root).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "omni-embed-nemotron-3b",
    )
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="nvomniembed",
        choices=MODEL_BACKBONE_CHOICES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step1_embeddings",
    )
    parser.add_argument(
        "--embedding-npz-name",
        type=str,
        default="embeddings_and_queries.npz",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all valid tuples")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--video-batch-size", type=int, default=1)
    parser.add_argument("--audio-batch-size", type=int, default=4)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--video-num-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tuples = read_mscoco_omni_tuples(
        dataset_root=args.dataset_root,
        max_samples=args.max_samples,
        meta_file=args.meta_file,
        catalog_file=args.catalog_file,
    )
    print(f"[INFO] Loaded valid semantic tuples: {len(tuples)}")

    directions = build_directions(MODALITY_ORDER)
    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading model from: {args.model_path}")
    model, processor, data_args = load_model_bundle(
        model_path=args.model_path,
        model_backbone=args.model_backbone,
        device=device,
    )

    semantic_ids, modalities, embeddings, query_map = build_embeddings_and_queries(
        tuples=tuples,
        model=model,
        processor=processor,
        model_backbone=args.model_backbone,
        data_args=data_args,
        device=device,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        video_batch_size=args.video_batch_size,
        audio_batch_size=args.audio_batch_size,
        audio_sample_rate=args.audio_sample_rate,
        video_num_frames=args.video_num_frames,
        directions=directions,
    )

    output_npz = args.output_dir / args.embedding_npz_name
    save_embedding_bundle(
        output_path=output_npz,
        semantic_ids=semantic_ids,
        modalities=modalities,
        embeddings=embeddings,
        query_map=query_map,
        directions=directions,
    )

    modality_counts = {m: int((modalities == m).sum()) for m in MODALITY_ORDER}
    summary = {
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_semantic_ids": int(len(set(semantic_ids.tolist()))),
        "modality_counts": modality_counts,
        "model_path": str(args.model_path),
        "model_backbone": args.model_backbone,
        "dataset_root": str(args.dataset_root),
        "meta_file": str(args.meta_file) if args.meta_file is not None else "auto",
        "catalog_file": str(args.catalog_file) if args.catalog_file is not None else "none",
        "embedding_npz": str(output_npz),
    }
    with (args.output_dir / "embedding_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Saved embedding bundle to: {output_npz}")


if __name__ == "__main__":
    main()
