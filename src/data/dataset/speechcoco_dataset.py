"""
SpeechCOCO 评测数据准备（音频 -> 图像检索）
- cand_image 来自 corpus parquet 的 image.bytes（不依赖图片根目录）
- query 不携带候选池，避免爆 CPU/内存
- corpus 返回框架期望 schema：['cand_text','cand_image','dataset_infos']
"""
import os
from typing import Any, Dict, List, Tuple

import datasets
from datasets import Features, Value

from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.model.processor import process_input_text


def _extract_audio_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    # query parquet schema: audio={"bytes","path"}
    audio = row.get("audio", None)
    if isinstance(audio, dict):
        return {"path": audio.get("path"), "bytes": audio.get("bytes")}
    return {"path": row.get("audio_path") or row.get("path"), "bytes": row.get("audio_bytes")}


def _image_to_cand_image_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    # corpus parquet schema: image={"bytes","path"}
    img = row.get("image", None)
    if not isinstance(img, dict):
        raise ValueError("[SpeechCOCO] corpus row['image'] is not a dict")
    b = img.get("bytes", None)
    if b is None:
        raise ValueError("[SpeechCOCO] corpus image.bytes is None")
    # 强制只走 bytes 解码：paths[0]=None
    return {"paths": [None], "bytes": [b], "resolutions": [None]}


def build_speechcoco_audio_image_dataset(path_info, **kwargs) -> Tuple[datasets.Dataset, Any]:
    dataset_path, subset, split = path_info

    model_backbone = kwargs.get("model_backbone", None)
    if model_backbone is None:
        raise ValueError("[SpeechCOCO] model_backbone is required")

    # -------- 1) load query/corpus（保留你的读取方式）--------
    query_dataset = None
    corpus_dataset = None

    if "-1k" in dataset_path:
        import glob
        query_files = sorted(glob.glob(os.path.join(dataset_path, "*query*.parquet")))
        corpus_files = sorted(glob.glob(os.path.join(dataset_path, "*corpus*.parquet")))
        if not query_files or not corpus_files:
            ds = load_hf_dataset(path_info)
            ds = sample_dataset(ds, **kwargs)
            query_dataset = ds
            corpus_dataset = None
        else:
            query_features = Features({
                "id": Value("string"),
                "audio": {"bytes": Value("binary"), "path": Value("string")},
                "image_id": Value("string"),
            })
            query_dataset = datasets.load_dataset(
                "parquet", data_files={"data": query_files}, split="data", features=query_features
            )

            corpus_features = Features({
                "corpus-id": Value("string"),
                "image": {"bytes": Value("binary"), "path": Value("string")},
            })
            corpus_dataset = datasets.load_dataset(
                "parquet", data_files={"data": corpus_files}, split="data", features=corpus_features
            )
    else:
        ds = load_hf_dataset(path_info)
        ds = sample_dataset(ds, **kwargs)
        query_dataset = ds
        corpus_dataset = None

    assert query_dataset is not None, "[SpeechCOCO] query_dataset is None"
    if corpus_dataset is None:
        raise ValueError("[SpeechCOCO] corpus_dataset is None (audio->image 必须要 corpus parquet)")

    # -------- 2) build corpus（框架期望 schema）--------
    # cand_text：必须含 <|image_pad|>（用 process_input_text 注入）
    cand_inst = process_input_text(
        "Understand the content of the provided image.",
        model_backbone,
        add_image_token=True,
    )

    # 用 map 构造，避免把 bytes 全部拉进 Python list 里反复复制
    def _corpus_map_fn(row, idx):
        cid = str(row["corpus-id"])
        return {
            "cand_text": [cand_inst],
            "cand_image": [_image_to_cand_image_obj(row)],
            "dataset_infos": {"cand_names": [cid], "corpus_id": cid, "label_name": cid},
        }

    corpus = corpus_dataset.map(_corpus_map_fn, with_indices=True, remove_columns=corpus_dataset.column_names)

    # 建 label 映射（1250 很小，这里在 Python 里建没问题）
    cand_names = [str(x) for x in corpus_dataset["corpus-id"]]
    # 去重保持顺序
    seen = set()
    uniq = []
    for x in cand_names:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    image_id2idx = {cid: i for i, cid in enumerate(uniq)}

    # -------- 3) build query dataset（不复制候选池，避免爆 CPU/内存）--------
    out_query_text, out_query_image, out_query_audio, out_query_audio_path = [], [], [], []
    out_cand_text, out_cand_image, out_infos = [], [], []

    for row in query_dataset:
        audio_obj = _extract_audio_obj(row)
        q_path = audio_obj.get("path") or row.get("path") or row.get("audio_path")

        img_id = row.get("image_id") or row.get("image") or row.get("image_name") or row.get("id")
        if img_id is None:
            raise ValueError("[SpeechCOCO] query missing image_id")
        img_id = str(img_id)

        label_idx = image_id2idx.get(img_id, -1)
        if label_idx < 0:
            raise ValueError(f"[SpeechCOCO] GT image_id not in corpus: {img_id}")

        out_query_text.append(build_query_text("SpeechCOCO"))  # query 侧没有 image，所以不用 image token
        out_query_image.append([None])
        out_query_audio.append(audio_obj)                      # ✅ dict（别包一层 list）
        out_query_audio_path.append(q_path)

        out_cand_text.append([])   # ✅ 不在 query 行复制候选池
        out_cand_image.append([])
        out_infos.append({
            "label_cand_id": label_idx,
            "label_name": img_id,
            "audio_id": row.get("id"),
            "image_id": img_id,
        })

    dataset = datasets.Dataset.from_dict({
        "query_text": out_query_text,
        "query_image": out_query_image,
        "query_audio": out_query_audio,
        "query_audio_path": out_query_audio_path,
        "cand_text": out_cand_text,
        "cand_image": out_cand_image,
        "dataset_infos": out_infos,
    })

    return dataset, corpus
