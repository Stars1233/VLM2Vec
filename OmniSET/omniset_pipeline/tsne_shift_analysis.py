"""t-SNE projection and cosine-shift analysis for OmniSET."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .constants import MODALITY_ORDER, SHORT_TO_SOURCE, build_directions
from .io_utils import l2_normalize


def run_tsne(
    embeddings: np.ndarray,
    perplexity: float,
    metric: str,
    random_state: int,
) -> Tuple[np.ndarray, float]:
    """Run 2D t-SNE and return coordinates with used perplexity."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be rank-2, got shape={emb.shape}")
    if emb.shape[0] < 2:
        raise ValueError("Need at least 2 samples for t-SNE")

    max_valid = float(max(1, emb.shape[0] - 1))
    used_perplexity = float(min(perplexity, max_valid))

    tsne = TSNE(
        n_components=2,
        perplexity=used_perplexity,
        metric=metric,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    xy = tsne.fit_transform(emb)
    return np.asarray(xy, dtype=np.float32), used_perplexity


def project_query_embeddings_to_tsne(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_xy: np.ndarray,
    k: int = 24,
    temperature: float = 0.07,
) -> np.ndarray:
    """Project query embeddings to background t-SNE space via top-k similarity interpolation."""
    query = np.asarray(query_embeddings, dtype=np.float32)
    ref = np.asarray(ref_embeddings, dtype=np.float32)
    ref_xy = np.asarray(ref_xy, dtype=np.float32)

    if query.ndim != 2 or ref.ndim != 2 or ref_xy.ndim != 2:
        raise ValueError("query/ref/ref_xy must be rank-2")
    if ref.shape[0] != ref_xy.shape[0]:
        raise ValueError("ref_embeddings and ref_xy row count mismatch")

    k = max(1, min(int(k), ref.shape[0]))
    query_norm = l2_normalize(query)
    ref_norm = l2_normalize(ref)

    sim = query_norm @ ref_norm.T
    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)

    order = np.argsort(topk_sim, axis=1)[:, ::-1]
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)
    topk_sim = np.take_along_axis(topk_sim, order, axis=1)

    logits = (topk_sim - topk_sim.max(axis=1, keepdims=True)) / max(float(temperature), 1e-6)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    topk_xy = ref_xy[topk_idx]
    proj = np.sum(weights[..., None] * topk_xy, axis=1)
    return np.asarray(proj, dtype=np.float32)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity for same-shape matrices."""
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        raise ValueError(f"Shape mismatch: {aa.shape} vs {bb.shape}")

    aa = l2_normalize(aa)
    bb = l2_normalize(bb)
    out = np.sum(aa * bb, axis=1)
    return np.clip(out, -1.0, 1.0)


def build_pair_shift_metrics(
    semantic_ids: np.ndarray,
    modalities: np.ndarray,
    embeddings: np.ndarray,
    query_map: Dict[str, np.ndarray],
    tsne_xy: np.ndarray,
    directions: Sequence[Tuple[str, str]] | None = None,
    query_shift_proj_k: int = 24,
) -> pd.DataFrame:
    """Build per-(query,target) shift metrics table with t-SNE coordinates and cosine deltas."""
    if directions is None:
        directions = build_directions(MODALITY_ORDER)

    sid = np.asarray(semantic_ids).astype(str)
    mod = np.asarray(modalities).astype(str)
    emb = np.asarray(embeddings, dtype=np.float32)
    xy = np.asarray(tsne_xy, dtype=np.float32)

    pair_to_idx = {(sid_i, mod_i): idx for idx, (sid_i, mod_i) in enumerate(zip(sid.tolist(), mod.tolist()))}

    rows: List[Dict[str, object]] = []
    for src, tgt in directions:
        key = f"{src}2{tgt}"
        q_instr_all = query_map.get(key)
        if q_instr_all is None:
            continue

        src_idx_all = np.where(mod == src)[0]
        n = min(int(src_idx_all.size), int(q_instr_all.shape[0]))
        if n <= 0:
            continue

        src_idx = np.asarray(src_idx_all[:n], dtype=np.int64)
        q_instr = np.asarray(q_instr_all[:n], dtype=np.float32)

        instr_xy = project_query_embeddings_to_tsne(
            query_embeddings=q_instr,
            ref_embeddings=emb,
            ref_xy=xy,
            k=query_shift_proj_k,
        )

        for local_i in range(n):
            s_idx = int(src_idx[local_i])
            sid_i = str(sid[s_idx])
            t_idx = pair_to_idx.get((sid_i, tgt))
            if t_idx is None:
                continue

            raw_emb = emb[s_idx : s_idx + 1]
            instr_emb = q_instr[local_i : local_i + 1]
            tgt_emb = emb[t_idx : t_idx + 1]

            cos_raw = float(cosine_similarity_rows(raw_emb, tgt_emb)[0])
            cos_instr = float(cosine_similarity_rows(instr_emb, tgt_emb)[0])
            delta_cos = float(cos_instr - cos_raw)

            raw_xy = xy[s_idx]
            ins_xy = instr_xy[local_i]
            tgt_xy = xy[t_idx]

            l2_raw = float(np.linalg.norm(raw_xy - tgt_xy))
            l2_instr = float(np.linalg.norm(ins_xy - tgt_xy))
            l2_delta = float(l2_raw - l2_instr)

            rows.append(
                {
                    "direction": key,
                    "source_modality": src,
                    "target_modality": tgt,
                    "query_index_in_direction": int(local_i),
                    "semantic_id": sid_i,
                    "source_index": s_idx,
                    "target_index": int(t_idx),
                    "raw_x": float(raw_xy[0]),
                    "raw_y": float(raw_xy[1]),
                    "instr_x": float(ins_xy[0]),
                    "instr_y": float(ins_xy[1]),
                    "target_x": float(tgt_xy[0]),
                    "target_y": float(tgt_xy[1]),
                    "cosine_sim_raw_to_target": cos_raw,
                    "cosine_sim_instr_to_target": cos_instr,
                    "delta_cosine_similarity_instr_minus_raw": delta_cos,
                    "proj_l2_raw_to_target": l2_raw,
                    "proj_l2_instr_to_target": l2_instr,
                    "proj_l2_improvement_raw_minus_instr": l2_delta,
                }
            )

    return pd.DataFrame(rows)


def build_cluster_point_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Build long-format points table for 7-cluster plotting."""
    if pair_df.empty:
        return pd.DataFrame(
            columns=[
                "source_modality",
                "plot_source_modality_name",
                "target_modality",
                "semantic_id",
                "point_role",
                "x",
                "y",
            ]
        )

    query_df = (
        pair_df.sort_values(["source_modality", "semantic_id", "target_modality"])
        .drop_duplicates(subset=["source_modality", "semantic_id"], keep="first")
        .copy()
    )
    query_df = query_df.assign(
        plot_source_modality_name=query_df["source_modality"].map(SHORT_TO_SOURCE),
        target_modality="none",
        point_role="query",
        x=query_df["raw_x"],
        y=query_df["raw_y"],
    )

    instr_df = pair_df.copy()
    instr_df = instr_df.assign(
        plot_source_modality_name=instr_df["source_modality"].map(SHORT_TO_SOURCE),
        point_role="instruction_query",
        x=instr_df["instr_x"],
        y=instr_df["instr_y"],
    )

    target_df = pair_df.copy()
    target_df = target_df.assign(
        plot_source_modality_name=target_df["source_modality"].map(SHORT_TO_SOURCE),
        point_role="target",
        x=target_df["target_x"],
        y=target_df["target_y"],
    )

    cols = [
        "source_modality",
        "plot_source_modality_name",
        "target_modality",
        "semantic_id",
        "point_role",
        "x",
        "y",
    ]
    out = pd.concat([query_df[cols], instr_df[cols], target_df[cols]], ignore_index=True)
    out["target_modality"] = out["target_modality"].fillna("none")
    return out


def compute_delta_matrix(
    pair_df: pd.DataFrame,
    modality_order: Sequence[str] = MODALITY_ORDER,
) -> pd.DataFrame:
    """Compute source-target matrix of mean delta cosine similarity."""
    mat = pd.DataFrame(np.nan, index=list(modality_order), columns=list(modality_order), dtype=float)
    for m in modality_order:
        mat.loc[m, m] = 0.0

    if pair_df.empty:
        return mat

    grouped = (
        pair_df.groupby(["source_modality", "target_modality"], as_index=False)[
            "delta_cosine_similarity_instr_minus_raw"
        ]
        .mean()
        .rename(columns={"delta_cosine_similarity_instr_minus_raw": "mean_delta_cosine"})
    )
    for row in grouped.itertuples(index=False):
        src = str(row.source_modality)
        tgt = str(row.target_modality)
        if src in mat.index and tgt in mat.columns:
            mat.loc[src, tgt] = float(row.mean_delta_cosine)
    return mat


def write_matrix_json(matrix: pd.DataFrame, output_path: Path) -> None:
    """Export matrix dataframe to JSON nested-dict format."""
    payload = {
        "metric": "mean_delta_cosine_similarity_instr_minus_raw",
        "note": "positive means instruction-conditioned query is closer to target",
        "matrix": {
            src: {
                tgt: (
                    None if not np.isfinite(float(matrix.loc[src, tgt])) else float(matrix.loc[src, tgt])
                )
                for tgt in matrix.columns
            }
            for src in matrix.index
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_analysis_summary(
    pair_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    tsne_metric: str,
    requested_perplexity: float,
    used_perplexity: float,
) -> Dict[str, object]:
    """Build compact summary for report JSON."""
    if pair_df.empty:
        overall = {
            "num_pairs": 0,
            "mean_cosine_sim_raw_to_target": None,
            "mean_cosine_sim_instr_to_target": None,
            "mean_delta_cosine_similarity_instr_minus_raw": None,
            "improved_ratio_cosine_similarity": None,
            "mean_proj_l2_raw_to_target": None,
            "mean_proj_l2_instr_to_target": None,
            "improved_ratio_proj_l2": None,
        }
    else:
        overall = {
            "num_pairs": int(len(pair_df)),
            "mean_cosine_sim_raw_to_target": float(pair_df["cosine_sim_raw_to_target"].mean()),
            "mean_cosine_sim_instr_to_target": float(pair_df["cosine_sim_instr_to_target"].mean()),
            "mean_delta_cosine_similarity_instr_minus_raw": float(
                pair_df["delta_cosine_similarity_instr_minus_raw"].mean()
            ),
            "improved_ratio_cosine_similarity": float(
                (pair_df["delta_cosine_similarity_instr_minus_raw"] > 0).mean()
            ),
            "mean_proj_l2_raw_to_target": float(pair_df["proj_l2_raw_to_target"].mean()),
            "mean_proj_l2_instr_to_target": float(pair_df["proj_l2_instr_to_target"].mean()),
            "improved_ratio_proj_l2": float((pair_df["proj_l2_improvement_raw_minus_instr"] > 0).mean()),
        }

    per_direction = []
    if not pair_df.empty:
        grouped = pair_df.groupby(["source_modality", "target_modality"], as_index=False)
        for g in grouped:
            (src, tgt), sub = g
            per_direction.append(
                {
                    "direction": f"{src}2{tgt}",
                    "source_modality": str(src),
                    "target_modality": str(tgt),
                    "count": int(len(sub)),
                    "mean_delta_cosine_similarity_instr_minus_raw": float(
                        sub["delta_cosine_similarity_instr_minus_raw"].mean()
                    ),
                    "improved_ratio_cosine_similarity": float(
                        (sub["delta_cosine_similarity_instr_minus_raw"] > 0).mean()
                    ),
                }
            )

    return {
        "overall": overall,
        "per_direction": per_direction,
        "matrix": {
            src: {
                tgt: (
                    None
                    if not np.isfinite(float(matrix_df.loc[src, tgt]))
                    else float(matrix_df.loc[src, tgt])
                )
                for tgt in matrix_df.columns
            }
            for src in matrix_df.index
        },
        "tsne": {
            "metric": tsne_metric,
            "requested_perplexity": float(requested_perplexity),
            "used_perplexity": float(used_perplexity),
        },
    }
