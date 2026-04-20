#!/usr/bin/env python3
"""Step 3: plot sampled 7-cluster t-SNE figures and cosine-shift heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omniset_pipeline.plotting import (  # noqa: E402
    plot_delta_heatmap,
    plot_tsne_clusters_by_source,
    sample_cluster_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE cluster figures and cosine-shift heatmap.")
    parser.add_argument(
        "--cluster-points-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step2_analysis" / "tsne_cluster_points.csv",
    )
    parser.add_argument(
        "--delta-matrix-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step2_analysis" / "delta_cosine_matrix.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "step3_figures",
    )
    parser.add_argument("--cluster-sample-ratio", type=float, default=0.25)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cluster_points = pd.read_csv(args.cluster_points_csv)
    sampled_points = sample_cluster_points(cluster_points, sample_ratio=args.cluster_sample_ratio)
    sampled_points.to_csv(args.output_dir / "sampled_tsne_cluster_points.csv", index=False)

    cluster_dir = args.output_dir / "tsne_clusters"
    cluster_paths = plot_tsne_clusters_by_source(
        points_df=cluster_points,
        output_dir=cluster_dir,
        sample_ratio=args.cluster_sample_ratio,
        figure_dpi=args.dpi,
    )

    matrix_df = pd.read_csv(args.delta_matrix_csv, index_col=0)
    matrix_df.index = matrix_df.index.astype(str)
    matrix_df.columns = matrix_df.columns.astype(str)

    plot_delta_heatmap(
        matrix_df=matrix_df,
        output_png=args.output_dir / "delta_cosine_heatmap.png",
        output_pdf=args.output_dir / "delta_cosine_heatmap.pdf",
        figure_dpi=args.dpi,
    )

    print(f"[INFO] Sampled cluster points: {len(sampled_points)}")
    print(f"[INFO] Saved cluster figures: {len(cluster_paths)}")
    for p in cluster_paths:
        print(f"[FIG] {p}")
    print(f"[FIG] {args.output_dir / 'delta_cosine_heatmap.png'}")


if __name__ == "__main__":
    main()
