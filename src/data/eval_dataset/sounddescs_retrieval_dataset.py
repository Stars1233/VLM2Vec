"""
SoundDescs 文本 -> 音频 检索评测数据集：
- 文本 -> 音频 检索数据处理（供评测路由调用）。
- query: text
- candidates: 全部音频池（同 split）
- label_cand_id: 正例音频在 cand_audio 中的索引
"""

import glob
import os
from typing import Any, Dict, List, Tuple, Optional

import datasets

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.utils.dataset_utils import load_qrels_mapping, sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text


# -------------------------
# ✅ 新 parquet 已经 flatten：
#   corpus: corpus-id, audio_path, audio_bytes (可能还有 id)
#   query : query-id + 原字段（text/caption/sentence/query/...）
#   qrels : query-id, corpus-id, score
# -------------------------


def _extract_audio_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    """处理嵌套audio结构或扁平列"""
    # 首先尝试从扁平列获取（预处理后的数据）
    if "audio_path" in row or "audio_bytes" in row:
        return {
            "path": row.get("audio_path") or row.get("path"),
            "bytes": row.get("audio_bytes"),
        }

    # 处理嵌套audio结构
    if "audio" in row and isinstance(row["audio"], dict):
        audio_struct = row["audio"]
        return {
            "path": audio_struct.get("path"),
            "bytes": audio_struct.get("bytes"),
        }

    # 回退到path字段
    return {
        "path": row.get("path"),
        "bytes": None,
    }


def _extract_text(row: Dict[str, Any]) -> str:
    """从 query 行里抽文本（字段名不固定）"""
    for key in ["text", "caption", "sentence", "query"]:
        v = row.get(key, None)
        if isinstance(v, str) and v.strip():
            return v
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError(f"No text field found. keys={list(row.keys())}")


def _load_parquet_stream(
    pattern: str,
    columns: Optional[List[str]] = None,
):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {pattern}")

    kw = dict(
        path="parquet",
        data_files={"data": files},
        streaming=True,
    )
    # 移除columns参数，让datasets自动处理嵌套结构
    # if columns is not None:
    #     kw["columns"] = columns

    return datasets.load_dataset(**kw)["data"]


def _prepare_corpus(dataset_path: str) -> Tuple[List[List[Dict[str, Any]]], Dict[str, int], List[str]]:
    # 兼容：sounddescs-1k 的数据在 eval/ 下（你的重建脚本就是这样）
    if "-1k" in dataset_path:
        corpus_pattern = os.path.join(dataset_path, "eval", "corpus*.parquet")
    else:
        corpus_pattern = os.path.join(dataset_path, "corpus", "*.parquet")

    # 读取所有列，包括嵌套的audio结构
    corpus_stream = _load_parquet_stream(
        corpus_pattern,
        columns=None,  # 读取所有列以处理嵌套结构
    )

    cand_audio_pool: List[List[Dict[str, Any]]] = []
    cand_names: List[str] = []

    for row in corpus_stream:
        audio_obj = _extract_audio_obj(row)
        # 优先使用corpus-id，然后是id，最后使用path
        cid = row.get("corpus-id") or row.get("id")
        if cid is None and audio_obj.get("path"):
            # 从path中提取文件名作为id
            # (REMOVE) do not import/rebind os inside this function; use the global import at file top
            cid = os.path.splitext(os.path.basename(audio_obj["path"]))[0]
        cid = str(cid)

        cand_names.append(cid)
        cand_audio_pool.append([audio_obj])

    cand_id2idx = {cid: idx for idx, cid in enumerate(cand_names)}
    return cand_audio_pool, cand_id2idx, cand_names


def _prepare_queries(dataset_path: str, **kwargs):
    if "-1k" in dataset_path:
        query_pattern = os.path.join(dataset_path, "eval", "query*.parquet")
    else:
        query_pattern = os.path.join(dataset_path, "query", "*.parquet")

    # query 不强制 columns：避免你文本字段名变化
    queries_stream = _load_parquet_stream(query_pattern, columns=None)

    # ✅ query 规模不大：stream -> list -> Dataset，然后 sample
    queries_data = list(queries_stream)
    queries_ds = datasets.Dataset.from_list(queries_data)
    queries_ds = sample_dataset(queries_ds, **kwargs)
    return queries_ds


def _load_qrels(dataset_path: str) -> Dict[str, Dict[str, int]]:
    if "-1k" in dataset_path:
        qrels_pattern = os.path.join(dataset_path, "eval", "qrels*.parquet")
    else:
        qrels_pattern = os.path.join(dataset_path, "qrels", "*.parquet")

    # ✅ 标准三列即可
    qrels_stream = _load_parquet_stream(qrels_pattern, columns=["query-id", "corpus-id", "score"])
    qrels_data = list(qrels_stream)
    qrels_ds = datasets.Dataset.from_list(qrels_data)
    return load_qrels_mapping(qrels_ds)


def _data_prepare(batch: Dict[str, List[Any]], **kwargs):
    """
    ✅ 这里一定要只基于 `batch` 来构造输出，每个输出列长度必须 == batch_size
    不能用全局 query_ids_batch / query_texts_batch，否则会出现长度错配（你之前的 256 vs 1000）
    """
    cand_audio_pool: List[List[Dict[str, Any]]] = kwargs["cand_audio_pool"]
    cand_id2idx: Dict[str, int] = kwargs["cand_id2idx"]
    cand_names: List[str] = kwargs["cand_names"]
    qrels: Dict[str, Dict[str, int]] = kwargs["qrels"]

    # ✅ 非空占位符：保证 tokenizer 至少有东西可编码
    cand_text_placeholder = ["[AUDIO]"] * len(cand_names)
    cand_image_placeholder = [None] * len(cand_names)

    batch_size = len(batch[next(iter(batch))])

    out_query_text, out_query_image, out_query_audio = [], [], []
    out_cand_text, out_cand_image, out_cand_audio, out_infos = [], [], [], []

    for i in range(batch_size):
        # 取 qid：你重建脚本写了 query-id
        qid = None
        for k in ["query-id", "query_id", "id", "qid"]:
            if k in batch:
                qid = batch[k][i]
                break
        if qid is None:
            raise ValueError(f"Query batch has no id fields. keys={list(batch.keys())}")
        qid = str(qid)

        # 还原这一行的 dict，用于抽文本
        row_i = {k: v[i] for k, v in batch.items()}
        raw_text = _extract_text(row_i)
        query_text = build_query_text("SoundDescs", raw_text)
        assert isinstance(query_text, list) and len(query_text) == 1 and isinstance(query_text[0], str) and query_text[0].strip()

        rels = qrels.get(qid, {})
        if rels:
            rel_cid = max(rels.items(), key=lambda x: x[1])[0]
            rel_cid = str(rel_cid)
            label_idx = cand_id2idx.get(rel_cid, -1)
        else:
            rel_cid = None
            label_idx = -1

        out_query_text.append(query_text)   # List[str] len=1
        out_query_image.append([None])
        out_query_audio.append(None)

        # ✅ 关键：cand_text/cand_image/cand_names 必须同长度，否则框架后面会炸
        out_cand_text.append(cand_text_placeholder)     # len == #cands
        out_cand_image.append(cand_image_placeholder)   # len == #cands
        out_cand_audio.append(cand_audio_pool)          # len == #cands

        out_infos.append({
            "label_cand_id": label_idx,
            "query_id": qid,
            "corpus_id": rel_cid,
            "cand_names": cand_names,  # ✅ 全量候选名（别再是 []）
            "label_name": rel_cid,  # ✅ 添加 label_name 字段，使用 corpus_id 作为标签
        })

    return {
        "query_text": out_query_text,
        "query_image": out_query_image,
        "query_audio": out_query_audio,
        "cand_text": out_cand_text,
        "cand_image": out_cand_image,
        "cand_audio": out_cand_audio,
        "dataset_infos": out_infos,
    }


def build_sounddescs_text_audio_dataset(path_info: Tuple[str, str, str], **kwargs):
    """
    返回 (dataset, corpus) 供评测使用。corpus 为空（候选池随样本携带）。
    """
    dataset_path = path_info[0]

    cand_audio_pool, cand_id2idx, cand_names = _prepare_corpus(dataset_path)
    qrels = _load_qrels(dataset_path)
    queries_ds = _prepare_queries(dataset_path, **kwargs)

    # ✅ 构造占位符，保证框架一致性检查通过
    cand_text_placeholder = ["[AUDIO]"] * len(cand_names)
    cand_image_placeholder = [None] * len(cand_names)

    map_kwargs = dict(kwargs)
    map_kwargs["cand_audio_pool"] = cand_audio_pool
    map_kwargs["cand_id2idx"] = cand_id2idx
    map_kwargs["cand_names"] = cand_names
    map_kwargs["cand_text_placeholder"] = cand_text_placeholder
    map_kwargs["cand_image_placeholder"] = cand_image_placeholder
    map_kwargs["qrels"] = qrels

    dataset = queries_ds.map(
        lambda batch: _data_prepare(batch, **map_kwargs),
        batched=True,
        batch_size=256,
        drop_last_batch=False,
        load_from_cache_file=False,
    )

    dataset = dataset.select_columns(
        [
            "query_text",
            "query_image",
            "query_audio",
            "cand_text",
            "cand_image",
            "cand_audio",
            "dataset_infos",
        ]
    )

    corpus = None
    return dataset, corpus


DATASET_PARSER_NAME = "sounddescs_text_audio"


DATASET_PARSER_NAME = "sounddescs_text_audio"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_sounddescs_text_audio_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "SoundDescs")
    path_info = EVAL_DATASET_LOCAL_PATH.get(dataset_name, EVAL_DATASET_HF_PATH.get(dataset_name))
    if path_info is None:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    return build_sounddescs_text_audio_dataset(path_info, **kwargs)