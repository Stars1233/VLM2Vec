#!/usr/bin/env python3
import os
import glob
import random
from typing import Dict, Any, List, Tuple

import datasets

# ✅ 新增：pyarrow 用于流式重写 corpus
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SRC_ROOT = "/code/.cache/datasets/MMEB-v2_1/audio-tasks/sounddescs"
OUT_DIR  = "/code/.cache/datasets/MMEB-v2_1/audio-tasks/sounddescs-1k"

N_EVAL = 1000
SEED = 17

# ✅ 控制 CPU / 内存：batch 越小越省（但会稍慢）
CORPUS_BATCH_SIZE = 256
PYARROW_USE_THREADS = False


def load_parquet_dir(dirpath: str) -> datasets.Dataset:
    files = sorted(glob.glob(os.path.join(dirpath, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet under: {dirpath}")
    return datasets.load_dataset("parquet", data_files={"data": files}, split="data")


def list_parquet_files(dirpath: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(dirpath, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet under: {dirpath}")
    return files


def pick_first_existing(cols: List[str], available: List[str], name: str) -> str:
    s = set(available)
    for c in cols:
        if c in s:
            return c
    raise KeyError(f"[{name}] cannot find any of {cols}. available={available}")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def dump_query(ds: datasets.Dataset, out_dir: str, qid_col: str):
    def _add(ex):
        ex["query-id"] = str(ex[qid_col])
        return ex
    ds2 = ds.map(_add, desc="add query-id", load_from_cache_file=False)
    ds2.to_parquet(os.path.join(out_dir, "query.parquet"))


def dump_qrels(ds: datasets.Dataset, out_dir: str, qid_col: str, qrels_by_qid: Dict[str, Tuple[str, float]]):
    rows = []
    for r in ds:
        qid = str(r[qid_col])
        pos_cid, _ = qrels_by_qid[qid]
        rows.append({"query-id": qid, "corpus-id": str(pos_cid), "score": 1})
    datasets.Dataset.from_list(rows).to_parquet(os.path.join(out_dir, "qrels.parquet"))


def rewrite_corpus_flat_pyarrow(
    corpus_files: List[str],
    out_path: str,
    cid_col: str,
):
    """
    ✅ 流式重写 corpus：
    - 不用 datasets.map（避免 CPU 爆）
    - 每次读取一小批 record_batch
    - audio(struct) -> audio_path/audio_bytes/(audio_array/audio_sampling_rate)
    - 删除 audio 列，保留其它列
    - 确保有 corpus-id 列（string）
    """
    writer = None

    for fp in corpus_files:
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=CORPUS_BATCH_SIZE, use_threads=PYARROW_USE_THREADS):
            t = pa.Table.from_batches([rb])

            # 1) 生成 corpus-id（如果原来就有则保持；否则用 cid_col）
            if "corpus-id" in t.column_names:
                corpus_id_arr = pc.cast(t["corpus-id"], pa.string())
            else:
                if cid_col not in t.column_names:
                    raise KeyError(f"cid_col={cid_col} not in columns: {t.column_names}")
                corpus_id_arr = pc.cast(t[cid_col], pa.string())
                t = t.append_column("corpus-id", corpus_id_arr)

            # 2) 展平 audio struct（如果存在）
            if "audio" in t.column_names:
                audio_col = t["audio"]
                # audio 必须是 struct 才能 field 提取；不是 struct 就跳过
                if pa.types.is_struct(audio_col.type):
                    fields = set(audio_col.type)
                    # path/bytes
                    if "path" in fields:
                        t = t.append_column("audio_path", pc.struct_field(audio_col, "path"))
                    if "bytes" in fields:
                        t = t.append_column("audio_bytes", pc.struct_field(audio_col, "bytes"))
                    # 可选字段：array/sampling_rate（有就保留）
                    if "array" in fields:
                        t = t.append_column("audio_array", pc.struct_field(audio_col, "array"))
                    if "sampling_rate" in fields:
                        t = t.append_column("audio_sampling_rate", pc.struct_field(audio_col, "sampling_rate"))
                # 删除 nested audio（关键：规避 nested bug）
                t = t.drop(["audio"])

            # 3) 写出（第一次初始化 writer）
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="snappy", use_dictionary=True)
            writer.write_table(t)

    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No data written for corpus (empty input?)")


def main():
    rng = random.Random(SEED)

    out_train = os.path.join(OUT_DIR, "train")
    out_eval  = os.path.join(OUT_DIR, "eval")
    ensure_dir(out_train)
    ensure_dir(out_eval)

    # 1) load query/qrels（小，OK）
    query_ds  = load_parquet_dir(os.path.join(SRC_ROOT, "query"))
    qrels_ds  = load_parquet_dir(os.path.join(SRC_ROOT, "qrels"))

    # 2) corpus 不用 datasets 全量 load（会重） -> 只拿文件列表 + 用 pyarrow 流式
    corpus_dir = os.path.join(SRC_ROOT, "corpus")
    corpus_files = list_parquet_files(corpus_dir)

    # 为了选 cid_col：只用 datasets 轻量读一次 schema（不读内容）
    # （也可以用 pyarrow 读 schema，但 datasets 你原来就有）
    corpus_schema_probe = datasets.load_dataset("parquet", data_files={"data": corpus_files[:1]}, split="data")
    cid_col = pick_first_existing(["corpus_id", "corpus-id", "id", "audio_id", "docid"], corpus_schema_probe.column_names, "corpus")

    qid_col = pick_first_existing(["query_id", "query-id", "id", "qid"], query_ds.column_names, "query")
    qrels_qid_col   = pick_first_existing(["query-id", "query_id", "qid", "id"], qrels_ds.column_names, "qrels")
    qrels_cid_col   = pick_first_existing(["corpus-id", "corpus_id", "cid", "docid", "id"], qrels_ds.column_names, "qrels")
    qrels_score_col = pick_first_existing(["score", "relevance", "rel"], qrels_ds.column_names, "qrels")

    # 3) qrels 映射
    qrels_by_qid: Dict[str, Tuple[str, float]] = {}
    for r in qrels_ds:
        qid = str(r[qrels_qid_col])
        cid = str(r[qrels_cid_col])
        sc  = float(r[qrels_score_col])
        if (qid not in qrels_by_qid) or (sc > qrels_by_qid[qid][1]):
            qrels_by_qid[qid] = (cid, sc)

    # 4) corpus id set：这里用 pyarrow 快速扫一遍“cid_col”列（避免全量 datasets load）
    corpus_ids = set()
    for fp in corpus_files:
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=2048, columns=[cid_col], use_threads=PYARROW_USE_THREADS):
            arr = rb.column(0)
            # to_pylist 对 10k 级别很安全；如果更大可改成逐个 as_py
            corpus_ids.update(str(x) for x in arr.to_pylist())

    # 5) 过滤可用 query
    kept_idx = []
    for i, r in enumerate(query_ds):
        qid = str(r[qid_col])
        if qid not in qrels_by_qid:
            continue
        pos_cid, _ = qrels_by_qid[qid]
        if pos_cid not in corpus_ids:
            continue
        kept_idx.append(i)

    if len(kept_idx) <= N_EVAL:
        raise RuntimeError(f"Not enough queries after filtering: {len(kept_idx)} <= N_EVAL={N_EVAL}")

    rng.shuffle(kept_idx)
    eval_idx  = kept_idx[:N_EVAL]
    train_idx = kept_idx[N_EVAL:]

    train_q = query_ds.select(train_idx)
    eval_q  = query_ds.select(eval_idx)

    # 6) 写 query/qrels
    dump_query(train_q, out_train, qid_col)
    dump_query(eval_q,  out_eval,  qid_col)

    dump_qrels(train_q, out_train, qid_col, qrels_by_qid)
    dump_qrels(eval_q,  out_eval,  qid_col, qrels_by_qid)

    # 7) 写 corpus（关键：pyarrow 流式 flatten，CPU 友好）
    rewrite_corpus_flat_pyarrow(
        corpus_files=corpus_files,
        out_path=os.path.join(out_train, "corpus.parquet"),
        cid_col=cid_col,
    )
    # train/eval 共用同一个
    # 直接复制文件（避免重复计算）
    import shutil
    shutil.copy2(os.path.join(out_train, "corpus.parquet"), os.path.join(out_eval, "corpus.parquet"))

    print("[DONE] SoundDescs-1k(flat) built at:", OUT_DIR)
    print(" - corpus written to:", os.path.join(out_eval, "corpus.parquet"))
    print(" - cid_col used:", cid_col)
    print(" - kept queries:", len(kept_idx), "eval:", len(eval_q), "train:", len(train_q))


if __name__ == "__main__":
    main()