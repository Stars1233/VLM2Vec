#!/usr/bin/env python3
import os
import glob
import random
import hashlib
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import datasets

# ========= CONFIG =========
SRC_ROOT = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/nsynth/data"
OUT_DIR  = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/nsynth-1k"

N_EVAL = 1000
SEED = 17

# NOTE: Comment translated to English.
MAX_CLASSES: Optional[int] = None  # e.g., 50 or 100, default None

# NOTE: Comment translated to English.
MIN_PER_CLASS_EVAL = 1


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_all_parquets(root: str) -> datasets.Dataset:
    # NOTE: Comment translated to English.
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {root}")
    return datasets.load_dataset("parquet", data_files={"data": files}, split="data")


def write_parquet(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        raise RuntimeError(f"Empty rows for {out_path}")
    keys = list(rows[0].keys())
    obj = {k: [r.get(k) for r in rows] for k in keys}
    datasets.Dataset.from_dict(obj).to_parquet(out_path)


def build_portable_audio_path(source_path: str, index: int) -> str:
    ext = os.path.splitext(source_path)[1].lower()
    if not ext:
        ext = ".wav"
    digest = hashlib.sha1((source_path or f"row-{index}").encode("utf-8")).hexdigest()[:10]
    return f"audio/{index:06d}-{digest}{ext}"


def to_portable_audio(audio_obj: Any, fallback_path: Optional[str], portable_path: str) -> Dict[str, Any]:
    raw_bytes = None
    source_path = fallback_path

    if isinstance(audio_obj, dict):
        raw_bytes = audio_obj.get("bytes")
        source_path = audio_obj.get("path") or fallback_path
    elif isinstance(audio_obj, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(audio_obj)
    elif isinstance(audio_obj, str):
        source_path = audio_obj

    if raw_bytes is None:
        if not source_path:
            raise ValueError("Cannot materialize audio bytes: no source path and no inline bytes")
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Audio file not found while materializing bytes: {source_path}")
        with open(source_path, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = bytes(raw_bytes)

    return {"path": portable_path, "bytes": raw_bytes}


def main():
    rng = random.Random(SEED)

    out_eval = os.path.join(OUT_DIR, "eval")
    ensure_dir(out_eval)

    ds = load_all_parquets(SRC_ROOT)

    # NOTE: Comment translated to English.
    if "WavPath" not in ds.column_names:
        raise KeyError(f"[nsynth] missing WavPath. available={ds.column_names}")

    # NOTE: Comment translated to English.
    label_col_candidates = ["instrument_family_str", "instrument_str", "instrument_family", "instrument"]
    label_col = None
    for c in label_col_candidates:
        if c in ds.column_names:
            label_col = c
            break
    if label_col is None:
        raise KeyError(f"[nsynth] cannot find label column in {label_col_candidates}. available={ds.column_names}")

    # NOTE: Comment translated to English.
    label_counts = defaultdict(int)
    for r in ds:
        lab = str(r[label_col])
        label_counts[lab] += 1

    labels_sorted = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))
    if MAX_CLASSES is not None:
        labels_sorted = labels_sorted[:MAX_CLASSES]

    labels_set = set(labels_sorted)

    # NOTE: Comment translated to English.
    # NOTE: Comment translated to English.
    per_label_indices = defaultdict(list)
    for i, r in enumerate(ds):
        lab = str(r[label_col])
        if lab in labels_set:
            per_label_indices[lab].append(i)

    # shuffle each label
    for lab in list(per_label_indices.keys()):
        rng.shuffle(per_label_indices[lab])

    eval_idx: List[int] = []
    # (a) min quota
    for lab in labels_sorted:
        pool = per_label_indices.get(lab, [])
        take = min(MIN_PER_CLASS_EVAL, len(pool))
        eval_idx.extend(pool[:take])
        per_label_indices[lab] = pool[take:]

        if len(eval_idx) >= N_EVAL:
            break

    if len(eval_idx) > N_EVAL:
        eval_idx = eval_idx[:N_EVAL]

    # (b) fill remaining
    if len(eval_idx) < N_EVAL:
        remaining = []
        for lab in labels_sorted:
            remaining.extend(per_label_indices.get(lab, []))
        rng.shuffle(remaining)
        need = N_EVAL - len(eval_idx)
        if len(remaining) < need:
            raise RuntimeError(f"Not enough data to fill eval to {N_EVAL}. remaining={len(remaining)}")
        eval_idx.extend(remaining[:need])

    rng.shuffle(eval_idx)
    eval_ds = ds.select(eval_idx)

    # NOTE: Comment translated to English.
    # NOTE: Comment translated to English.
    labels_final = sorted({str(r[label_col]) for r in eval_ds})
    label2id = {lab: i for i, lab in enumerate(labels_final)}

    corpus_rows = []
    for lab, cid in label2id.items():
        corpus_rows.append({
            "corpus-id": str(cid),
            "label": lab,
            "text": lab,  # NOTE: Comment translated to English.
        })

    # NOTE: Comment translated to English.
    # NOTE: Comment translated to English.
    query_rows = []
    qrels_rows = []

    for idx, r in enumerate(eval_ds):
        wav_path = str(r["WavPath"])
        qid = os.path.basename(wav_path)  # e.g., "acoustic_guitar_000-123.wav"
        lab = str(r[label_col])
        cid = str(label2id[lab])
        portable_audio_path = build_portable_audio_path(wav_path, idx)
        portable_audio = to_portable_audio(r.get("audio"), wav_path, portable_audio_path)

        # NOTE: Comment translated to English.
        query_rows.append({
            "query-id": qid,
            "WavPath": portable_audio["path"],
            "audio": portable_audio,  # NOTE: Comment translated to English.
            "label": lab,
            "label_id": int(cid),
        })

        # NOTE: Comment translated to English.
        qrels_rows.append({
            "query-id": qid,
            "corpus-id": cid,
            "score": 1,
        })

    # NOTE: Comment translated to English.
    write_parquet(query_rows, os.path.join(out_eval, "query.parquet"))
    write_parquet(qrels_rows, os.path.join(out_eval, "qrels.parquet"))
    write_parquet(corpus_rows, os.path.join(out_eval, "corpus_labels.parquet"))

    print("[DONE] NSynth-1k (eval-only) built at:", OUT_DIR)
    print(f" - eval queries: {len(query_rows)} (target={N_EVAL})")
    print(f" - labels(candidates): {len(corpus_rows)} (<=1w always for classification)")
    print(f" - label_col used: {label_col}")
    print(" - files:")
    print(f"   * {out_eval}/query.parquet")
    print(f"   * {out_eval}/qrels.parquet")
    print(f"   * {out_eval}/corpus_labels.parquet")


if __name__ == "__main__":
    main()
