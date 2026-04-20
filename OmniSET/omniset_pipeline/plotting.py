"""Plotting helpers for OmniSET t-SNE clusters and heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .constants import MODALITY_COLOR, MODALITY_NAME, MODALITY_ORDER, SHORT_TO_SOURCE, SOURCE_TO_SHORT


def sample_cluster_points(
    points_df: pd.DataFrame,
    sample_ratio: float = 0.25,
) -> pd.DataFrame:
    """Deterministically sample each cluster group by taking head(ratio*n)."""
    if points_df.empty:
        return points_df.copy()

    ratio = float(np.clip(sample_ratio, 0.0, 1.0))
    df = points_df.copy()
    df["target_modality"] = df["target_modality"].fillna("none")

    sort_cols = ["plot_source_modality_name", "point_role", "target_modality", "semantic_id"]
    for c in sort_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column for sampling: {c}")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    def _take_head(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        if ratio <= 0:
            keep = 1
        else:
            keep = max(1, int(n * ratio))
        return group.iloc[:keep]

    return (
        df.groupby(["plot_source_modality_name", "point_role", "target_modality"], group_keys=False)
        .apply(_take_head)
        .reset_index(drop=True)
    )


def _build_legend(alpha_query: float, alpha_inst: float, alpha_target: float) -> List[Line2D]:
    return [
        Line2D([0], [0], marker="s", color=MODALITY_COLOR["t"], linestyle="None", markersize=8, label="Text"),
        Line2D([0], [0], marker="s", color=MODALITY_COLOR["i"], linestyle="None", markersize=8, label="Image"),
        Line2D([0], [0], marker="s", color=MODALITY_COLOR["v"], linestyle="None", markersize=8, label="Video"),
        Line2D([0], [0], marker="s", color=MODALITY_COLOR["a"], linestyle="None", markersize=8, label="Audio"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=8, alpha=alpha_query, label="Query"),
        Line2D([0], [0], marker="v", color="black", linestyle="None", markersize=8, alpha=alpha_inst, label="Instruction + Query"),
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=8, alpha=alpha_target, label="Target"),
    ]


def plot_tsne_clusters_by_source(
    points_df: pd.DataFrame,
    output_dir: Path,
    sample_ratio: float = 0.25,
    figure_dpi: int = 300,
) -> List[Path]:
    """Plot one 7-cluster t-SNE figure for each source modality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = sample_cluster_points(points_df, sample_ratio=sample_ratio)
    if sampled.empty:
        return []

    x_min, x_max = float(sampled["x"].min()), float(sampled["x"].max())
    y_min, y_max = float(sampled["y"].min()), float(sampled["y"].max())
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    query_alpha = 0.9
    inst_alpha = 0.45
    target_alpha = 0.9

    produced: List[Path] = []
    for source_long in SHORT_TO_SOURCE.values():
        source_short = SOURCE_TO_SHORT[source_long]
        sub = sampled[sampled["plot_source_modality_name"] == source_long].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.0, 7.0))

        query_df = sub[sub["point_role"] == "query"]
        if not query_df.empty:
            ax.scatter(
                query_df["x"],
                query_df["y"],
                s=50,
                c=MODALITY_COLOR[source_long],
                marker="o",
                alpha=query_alpha,
                edgecolors="none",
                zorder=2,
            )

        inst_df = sub[sub["point_role"] == "instruction_query"]
        target_df = sub[sub["point_role"] == "target"]

        for tgt_short in MODALITY_ORDER:
            if tgt_short == source_short:
                continue
            inst_g = inst_df[inst_df["target_modality"] == tgt_short]
            if not inst_g.empty:
                ax.scatter(
                    inst_g["x"],
                    inst_g["y"],
                    s=50,
                    c=MODALITY_COLOR[tgt_short],
                    marker="v",
                    alpha=inst_alpha,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )

            tgt_g = target_df[target_df["target_modality"] == tgt_short]
            if not tgt_g.empty:
                ax.scatter(
                    tgt_g["x"],
                    tgt_g["y"],
                    s=50,
                    c=MODALITY_COLOR[tgt_short],
                    marker="^",
                    alpha=target_alpha,
                    edgecolors="none",
                    zorder=1,
                )

        ax.set_title(f"Source Query Modality: {MODALITY_NAME[source_long]}", fontsize=18)
        ax.tick_params(labelsize=11)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.legend(
            handles=_build_legend(query_alpha, inst_alpha, target_alpha),
            loc="upper left",
            fontsize=10,
            frameon=False,
            ncol=2,
        )

        out_path = output_dir / f"tsne_clusters_source_{source_long}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)
        produced.append(out_path)

    return produced


def plot_delta_heatmap(
    matrix_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path | None = None,
    figure_dpi: int = 300,
) -> None:
    """Plot source-target heatmap of mean delta cosine similarity."""
    labels = [MODALITY_NAME[m] for m in matrix_df.index.tolist()]
    values = matrix_df.to_numpy(dtype=float)

    abs_max = float(np.nanmax(np.abs(values[np.isfinite(values)]))) if np.isfinite(values).any() else 1.0
    if abs_max <= 0:
        abs_max = 1.0

    cmap = LinearSegmentedColormap.from_list("blue_gray_red", ["#3b6dcc", "#d9dce3", "#ff6b6b"])

    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=150)
    im = ax.imshow(values, cmap=cmap, vmin=-abs_max, vmax=abs_max)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)

    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_xlabel("Target Modality", fontsize=16, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Source Modality", fontsize=16, labelpad=8)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#aeb8c5", linestyle="-", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_color("#aeb8c5")
        spine.set_linewidth(1.5)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if i == j or not np.isfinite(values[i, j]):
                txt = ""
            else:
                txt = f"{values[i, j]:+.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=16, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.08)
    cbar.set_ticks([-abs_max, 0, abs_max])
    cbar.set_ticklabels([f"{-abs_max:+.3f}", "0", f"{abs_max:+.3f}"])
    cbar.outline.set_edgecolor("#9ca3af")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(labelsize=11, length=0, colors="#4b5563")

    plt.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=figure_dpi, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
