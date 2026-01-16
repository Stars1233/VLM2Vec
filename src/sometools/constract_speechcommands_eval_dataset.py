#!/usr/bin/env python3
import os
import glob
import random
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional, Set

import datasets

# ========= CONFIG =========
SRC_ROOT = "/code/.cache/datasets/MMEB-v2_1/audio-tasks/speechcommand"
OUT_DIR  = "/code/.cache/datasets/MMEB-v2_1/audio-tasks/speechcommand-1k"

N_TRAIN = 10_000
N_EVAL  = 1_000
SEED = 17

# 候选(label)控制：None=保留全部；否则只保留出现频率最高的前 K 个类
MAX_CLASSES: Optional[int] = None  # 例如 20；默认 None

# 每类至少在 eval 里保留的最少样本数（避免某类 eval 消失）
MIN_PER_CLASS_EVAL  = 1
# train 不强制每类都出现（因为 train 很大，一般都会出现），但你也可以设为 1
MIN_PER_CLASS_TRAIN = 0

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_parquet(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        raise RuntimeError(f"Empty rows for {out_path}")
    keys = list(rows[0].keys())
    obj = {k: [r.get(k) for r in rows] for k in keys}
    datasets.Dataset.from_dict(obj).to_parquet(out_path)


def list_audio_files_by_label(root: str) -> Dict[str, List[str]]:
    """
    SpeechCommands 常见结构：root/<label>/*.wav
    排除 _background_noise_ 以及隐藏目录。
    """
    label2files = defaultdict(list)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"SRC_ROOT not found: {root}")

    for label in sorted(os.listdir(root)):
        if label.startswith("."):
            continue
        if label == "_background_noise_":
            continue
        lab_dir = os.path.join(root, label)
        if not os.path.isdir(lab_dir):
            continue

        files = []
        for ext in AUDIO_EXTS:
            files.extend(glob.glob(os.path.join(lab_dir, f"*{ext}")))
        if files:
            # 去重 + 排序
            uniq = sorted(set(files))
            label2files[label].extend(uniq)

    return label2files


def pick_labels(label2files: Dict[str, List[str]], max_classes: Optional[int]) -> List[str]:
    counts = {lab: len(fs) for lab, fs in label2files.items()}
    labs = sorted(counts.keys(), key=lambda x: (-counts[x], x))
    if max_classes is not None:
        labs = labs[:max_classes]
    labs = [l for l in labs if counts.get(l, 0) > 0]
    if not labs:
        raise RuntimeError("No labels found after filtering")
    return labs


def stratified_disjoint_split(
    label2files: Dict[str, List[str]],
    labels: List[str],
    n_train: int,
    n_eval: int,
    rng: random.Random,
    min_train: int,
    min_eval: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    输出 disjoint 的 (train_pairs, eval_pairs)，pair=(filepath,label)

    步骤：
      1) 每类先分配 eval 的 min_eval，保证 eval 覆盖
      2) 再分配 train 的 min_train（可为0）
      3) 用剩余样本补齐 eval 到 n_eval
      4) 再用剩余样本补齐 train 到 n_train
    """
    pool = {lab: list(label2files[lab]) for lab in labels}
    for lab in labels:
        rng.shuffle(pool[lab])

    train: List[Tuple[str, str]] = []
    eval_: List[Tuple[str, str]] = []

    # 1) eval 基础覆盖
    for lab in labels:
        take_e = min(min_eval, len(pool[lab]))
        for _ in range(take_e):
            eval_.append((pool[lab].pop(), lab))

    # 2) train 基础覆盖（可选）
    for lab in labels:
        take_t = min(min_train, len(pool[lab]))
        for _ in range(take_t):
            train.append((pool[lab].pop(), lab))

    if len(eval_) > n_eval:
        raise RuntimeError(
            f"Eval min quota too large: eval={len(eval_)} > target={n_eval}. "
            f"Reduce MIN_PER_CLASS_EVAL or MAX_CLASSES."
        )

    # 3) 补齐 eval
    remaining = []
    for lab in labels:
        remaining.extend([(fp, lab) for fp in pool[lab]])
    rng.shuffle(remaining)

    while len(eval_) < n_eval and remaining:
        eval_.append(remaining.pop())

    if len(eval_) < n_eval:
        raise RuntimeError(
            f"Not enough data to fill eval target={n_eval}. current={len(eval_)}. "
            f"Try reduce N_EVAL or MAX_CLASSES."
        )

    # 4) 补齐 train（用 eval 之后剩下的）
    while len(train) < n_train and remaining:
        train.append(remaining.pop())

    if len(train) < n_train:
        raise RuntimeError(
            f"Not enough data to fill train target={n_train}. current={len(train)}. "
            f"Try reduce N_TRAIN or MAX_CLASSES."
        )

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def build_label_candidates(labels: List[str]) -> List[Dict[str, Any]]:
    # 分类任务 candidates = labels
    return [{"corpus-id": str(i), "label": lab, "text": lab} for i, lab in enumerate(labels)]


def build_queries(pairs: List[Tuple[str, str]], label2id: Dict[str, int], split: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, (fp, lab) in enumerate(pairs):
        qid = f"{split}-{idx:06d}"
        rows.append({
            "query-id": qid,
            "audio": {"path": fp, "bytes": None},
            "audio_path": fp,
            "label": lab,
            "label_id": int(label2id[lab]),
            "split": split,
        })
    return rows


def build_qrels(pairs: List[Tuple[str, str]], label2id: Dict[str, int], split: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, (_, lab) in enumerate(pairs):
        qid = f"{split}-{idx:06d}"
        rows.append({
            "query-id": qid,
            "corpus-id": str(label2id[lab]),
            "score": 1,
        })
    return rows


def main():
    rng = random.Random(SEED)
    ensure_dir(OUT_DIR)

    # 1) 扫描音频文件
    label2files = list_audio_files_by_label(SRC_ROOT)
    if not label2files:
        raise RuntimeError(f"No label folders with audio found under: {SRC_ROOT}")

    # 2) label 候选控制
    labels = pick_labels(label2files, MAX_CLASSES)

    # 3) disjoint stratified split：train=10k, eval=1k
    train_pairs, eval_pairs = stratified_disjoint_split(
        label2files=label2files,
        labels=labels,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        rng=rng,
        min_train=MIN_PER_CLASS_TRAIN,
        min_eval=MIN_PER_CLASS_EVAL,
    )

    # 4) candidates(labels)
    candidates = build_label_candidates(labels)
    label2id = {c["label"]: int(c["corpus-id"]) for c in candidates}

    # 5) queries / qrels
    train_queries = build_queries(train_pairs, label2id, "train")
    eval_queries  = build_queries(eval_pairs,  label2id, "eval")
    train_qrels   = build_qrels(train_pairs, label2id, "train")
    eval_qrels    = build_qrels(eval_pairs,  label2id, "eval")

    # 6) 写文件
    write_parquet(candidates,    os.path.join(OUT_DIR, "corpus_labels.parquet"))
    write_parquet(train_queries, os.path.join(OUT_DIR, "query_train.parquet"))
    write_parquet(eval_queries,  os.path.join(OUT_DIR, "query_eval.parquet"))
    write_parquet(train_qrels,   os.path.join(OUT_DIR, "qrels_train.parquet"))
    write_parquet(eval_qrels,    os.path.join(OUT_DIR, "qrels_eval.parquet"))

    # 7) 统计
    def stat(pairs):
        cnt = Counter([lab for _, lab in pairs])
        return len(pairs), len(cnt), cnt.most_common(5)

    ntr, ntr_cls, top_tr = stat(train_pairs)
    nev, nev_cls, top_ev = stat(eval_pairs)

    # 确认 train/eval 音频不重叠
    train_set = set(fp for fp, _ in train_pairs)
    eval_set  = set(fp for fp, _ in eval_pairs)
    inter = len(train_set & eval_set)

    print("[DONE] SpeechCommands built at:", OUT_DIR)
    print(f" - labels(candidates): {len(labels)} (MAX_CLASSES={MAX_CLASSES})")
    print(f" - train: q={ntr} | classes_present={ntr_cls} | top5={top_tr}")
    print(f" - eval : q={nev} | classes_present={nev_cls} | top5={top_ev}")
    print(f" - train/eval audio overlap: {inter} (should be 0)")
    print(" - files:")
    print("    corpus_labels.parquet")
    print("    query_train.parquet / qrels_train.parquet")
    print("    query_eval.parquet  / qrels_eval.parquet")


if __name__ == "__main__":
    main()