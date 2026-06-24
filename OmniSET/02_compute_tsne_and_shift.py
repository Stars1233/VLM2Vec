#!/usr/bin/env python3
"""Step 2: compute t-SNE coordinates and cosine-shift metrics from embedding bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omniset_pipeline.constants import MODALITY_ORDER, MODALITY_NAME, build_directions  # noqa: E402
from omniset_pipeline.io_utils import load_embedding_bundle  # noqa: E402
from omniset_pipeline.tsne_shift_analysis import (  # noqa: E402
    build_analysis_summary,
    build_cluster_point_table,
    build_pair_shift_metrics,
    compute_delta_matrix,
    run_tsne,
    write_matrix_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute t-SNE + cosine shift metrics from saved embeddings.")
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step1_embeddings" / "embeddings_and_queries.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step2_analysis",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-random-state", type=int, default=42)
    parser.add_argument("--tsne-metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--query-shift-proj-k", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    directions = build_directions(MODALITY_ORDER)
    semantic_ids, modalities, embeddings, query_map = load_embedding_bundle(
        input_path=args.embedding_npz,
        directions=directions,
    )

    print(f"[INFO] Loaded embedding bundle: {args.embedding_npz}")
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    tsne_xy, used_perplexity = run_tsne(
        embeddings=embeddings,
        perplexity=args.tsne_perplexity,
        metric=args.tsne_metric,
        random_state=args.tsne_random_state,
    )

    reference_df = pd.DataFrame(
        {
            "index": np.arange(len(semantic_ids), dtype=np.int64),
            "semantic_id": semantic_ids.astype(str),
            "modality": modalities.astype(str),
            "modality_name": [MODALITY_NAME.get(str(m), str(m)) for m in modalities],
            "tsne_x": tsne_xy[:, 0],
            "tsne_y": tsne_xy[:, 1],
        }
    )
    reference_df.to_csv(args.output_dir / "tsne_reference_points.csv", index=False)

    pair_df = build_pair_shift_metrics(
        semantic_ids=semantic_ids,
        modalities=modalities,
        embeddings=embeddings,
        query_map=query_map,
        tsne_xy=tsne_xy,
        directions=directions,
        query_shift_proj_k=args.query_shift_proj_k,
    )
    pair_df.to_csv(args.output_dir / "pair_shift_metrics.csv", index=False)

    cluster_df = build_cluster_point_table(pair_df)
    cluster_df.to_csv(args.output_dir / "tsne_cluster_points.csv", index=False)

    matrix_df = compute_delta_matrix(pair_df, modality_order=MODALITY_ORDER)
    matrix_df.index.name = "source_modality"
    matrix_df.to_csv(args.output_dir / "delta_cosine_matrix.csv")
    write_matrix_json(matrix_df, args.output_dir / "delta_cosine_matrix.json")

    summary = build_analysis_summary(
        pair_df=pair_df,
        matrix_df=matrix_df,
        tsne_metric=args.tsne_metric,
        requested_perplexity=args.tsne_perplexity,
        used_perplexity=used_perplexity,
    )
    summary["input_embedding_npz"] = str(args.embedding_npz)
    summary["outputs"] = {
        "tsne_reference_points_csv": str(args.output_dir / "tsne_reference_points.csv"),
        "pair_shift_metrics_csv": str(args.output_dir / "pair_shift_metrics.csv"),
        "tsne_cluster_points_csv": str(args.output_dir / "tsne_cluster_points.csv"),
        "delta_cosine_matrix_csv": str(args.output_dir / "delta_cosine_matrix.csv"),
        "delta_cosine_matrix_json": str(args.output_dir / "delta_cosine_matrix.json"),
    }

    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] t-SNE used perplexity: {used_perplexity:.2f}")
    print(f"[INFO] Pair rows: {len(pair_df)}")
    print(f"[INFO] Cluster point rows: {len(cluster_df)}")
    print(f"[INFO] Saved analysis outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
