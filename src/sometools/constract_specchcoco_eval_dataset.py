#!/usr/bin/env python3
import os
import glob
import random
from typing import List, Dict, Any

import datasets

SRC_DIR = "/code/.cache/datasets/MMEB-v2_1/audio-tasks/speechcoco/data"
OUT_DIR = "/code/.cache/datasets/MMEB-v2_1/audio-tasks/speechcoco-1k"

N_TRAIN = 1000
N_EVAL = 1000
CAND_MAX = 10000
SEED = 17

def _find_files() -> List[str]:
    files = sorted(glob.glob(os.path.join(SRC_DIR, "validation-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet found under: {SRC_DIR}")
    return files

def _infer_image_col(ds: datasets.Dataset) -> str:
    candidates = ["image", "image_path", "image_file", "img", "img_path"]
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Cannot find image column in dataset columns={sorted(cols)[:50]}...")

def _require_cols(ds: datasets.Dataset, cols: List[str]):
    missing = [c for c in cols if c not in ds.column_names]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available={ds.column_names}")

def _write_parquet(obj: Dict[str, list], path: str):
    datasets.Dataset.from_dict(obj).to_parquet(path)

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = _find_files()
    ds = datasets.load_dataset("parquet", data_files={"data": files}, split="data")

    _require_cols(ds, ["id", "image_id", "audio"])
    image_col = _infer_image_col(ds)
    print(f"[INFO] total rows={len(ds)} | image_col={image_col}")

    # 1) 收集 unique image_id -> image object（取第一次出现）
    img_map: Dict[str, Any] = {}
    for row in ds:
        img_id = str(row["image_id"])
        if img_id not in img_map:
            img_map[img_id] = row[image_col]
    all_img_ids = list(img_map.keys())
    random.shuffle(all_img_ids)
    print(f"[INFO] unique image_id={len(all_img_ids)}")

    # 2) 先粗分 train/eval candidate（尽量不重叠）
    #    这里直接按 image_id 分割，比按 row 分割更干净
    need_imgs = min(len(all_img_ids), 2 * CAND_MAX)
    train_img_ids = all_img_ids[: min(CAND_MAX, need_imgs)]
    eval_img_ids = all_img_ids[min(CAND_MAX, need_imgs): min(2 * CAND_MAX, need_imgs)]

    if len(eval_img_ids) < min(CAND_MAX, len(all_img_ids) - len(train_img_ids)):
        print(f"[WARN] not enough unique images to make eval_corpus fully disjoint at size {CAND_MAX}. "
              f"eval_corpus size will be {len(eval_img_ids)}.")

    train_cand_set = set(train_img_ids)
    eval_cand_set = set(eval_img_ids)

    overlap = len(train_cand_set & eval_cand_set)
    if overlap > 0:
        print(f"[WARN] candidate overlap detected: {overlap} images. (Should be 0 normally)")

    print(f"[INFO] corpus_train={len(train_img_ids)} | corpus_eval={len(eval_img_ids)}")

    # 3) 从原始 rows 中筛出“正例在对应候选池内”的 rows，再各自抽 1k
    train_rows = []
    eval_rows = []
    for row in ds:
        img_id = str(row["image_id"])
        if img_id in train_cand_set:
            train_rows.append(row)
        elif img_id in eval_cand_set:
            eval_rows.append(row)

    random.shuffle(train_rows)
    random.shuffle(eval_rows)

    if len(train_rows) < N_TRAIN:
        raise RuntimeError(f"Not enough train rows after filtering: {len(train_rows)} < {N_TRAIN}")
    if len(eval_rows) < N_EVAL:
        raise RuntimeError(f"Not enough eval rows after filtering: {len(eval_rows)} < {N_EVAL}")

    train_rows = train_rows[:N_TRAIN]
    eval_rows = eval_rows[:N_EVAL]

    # 4) 写 corpus（各自独立）
    _write_parquet(
        {"corpus-id": train_img_ids, "image": [img_map[iid] for iid in train_img_ids]},
        os.path.join(OUT_DIR, "corpus_train_10k.parquet"),
    )
    _write_parquet(
        {"corpus-id": eval_img_ids, "image": [img_map[iid] for iid in eval_img_ids]},
        os.path.join(OUT_DIR, "corpus_eval_10k.parquet"),
    )
    print("[OK] wrote corpus_train_10k.parquet and corpus_eval_10k.parquet")

    def build_query_and_qrels(rows: List[Dict[str, Any]], split: str):
        q = {
            "id": [str(r["id"]) for r in rows],
            "audio": [r["audio"] for r in rows],
            "image_id": [str(r["image_id"]) for r in rows],
        }
        _write_parquet(q, os.path.join(OUT_DIR, f"query_{split}.parquet"))

        qrels = {
            "query-id": [str(r["id"]) for r in rows],
            "corpus-id": [str(r["image_id"]) for r in rows],
            "score": [1 for _ in rows],
        }
        _write_parquet(qrels, os.path.join(OUT_DIR, f"qrels_{split}.parquet"))
        print(f"[OK] wrote query_{split}.parquet and qrels_{split}.parquet")

    build_query_and_qrels(train_rows, "train")
    build_query_and_qrels(eval_rows, "eval")

    print("\n[DONE] SpeechCOCO lite (disjoint candidates) built at:", OUT_DIR)
    print(f" - train: q={N_TRAIN}, c={len(train_img_ids)}")
    print(f" - eval : q={N_EVAL}, c={len(eval_img_ids)}")

if __name__ == "__main__":
    main()