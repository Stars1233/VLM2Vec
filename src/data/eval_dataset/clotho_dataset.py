"""
Clotho 文本 -> 音频 检索评测数据集：
- 文本 -> 音频检索数据处理
"""

import os
from typing import Any, Dict, List, Tuple

import datasets
import pandas as pd
from src.utils.basic_utils import print_master
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset


def _expand_clotho_rows(csv_path: str, audio_dir: str) -> List[Dict[str, str]]:
    """将 Clotho CSV 每行展开为 5 条 (text, audio_path)。"""
    df = pd.read_csv(csv_path)
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        file_name = row["file_name"]
        audio_path = os.path.join(audio_dir, file_name)
        for i in range(1, 6):
            cap = row.get(f"caption_{i}")
            if not isinstance(cap, str):
                continue
            records.append(
                {
                    "text": cap.strip(),
                    "audio_path": audio_path,
                    "file_name": file_name,
                }
            )
    return records


def build_clotho_text_audio_dataset(path_info: Tuple[str, str, str], **kwargs):
    """
    返回 (query_dataset, corpus_dataset)，其中：
    - query_dataset: 只包含查询信息和标签
    - corpus_dataset: 包含所有候选，每个条目一个候选
    """
    data_path = path_info[0]
    captions_csv: str = kwargs.get("captions_csv", "clotho_captions_evaluation.csv")
    audio_subdir: str = kwargs.get("audio_subdir", "evaluation")

    csv_path = os.path.join(data_path, captions_csv)
    audio_dir = os.path.join(data_path, audio_subdir)
    assert os.path.isfile(csv_path), f"未找到 caption CSV: {csv_path}"
    assert os.path.isdir(audio_dir), f"未找到音频目录: {audio_dir}"

    records = _expand_clotho_rows(csv_path, audio_dir)
    # 构造候选池与映射
    file_names = [r["file_name"] for r in records]
    unique_files = list(dict.fromkeys(file_names))  # 保持顺序去重
    file2idx = {fn: i for i, fn in enumerate(unique_files)}

    texts = [r["text"] for r in records]
    label_ids = [file2idx[r["file_name"]] for r in records]

    # 构建query_text并添加debug assert
    query_texts = []
    for t in texts:
        query_text = build_query_text("Clotho", t)
        assert isinstance(query_text, list) and len(query_text) == 1 and isinstance(query_text[0], str) and query_text[0].strip()
        query_texts.append(query_text)

    # 构建query_dataset - 只包含查询信息和标签
    query_dataset = datasets.Dataset.from_dict({
        "query_text": query_texts,
        "query_image": [[None] for _ in texts],
        "query_audio": [None for _ in texts],
        "dataset_infos": [{"label_cand_id": lid, "file_name": fn, "label_name": fn} for lid, fn in zip(label_ids, file_names)],
    })

    # 构建corpus_dataset - 包含所有候选，每个条目一个候选
    corpus_rows = []
    for fn in unique_files:
        audio_obj = {"path": os.path.join(audio_dir, fn), "bytes": None}
        corpus_rows.append({
            "cand_text": ["[AUDIO]"],  # 音频候选的文本占位符
            "cand_image": [None],
            "cand_audio": [audio_obj],
            "dataset_infos": {"cand_names": [fn]},  # 用于 cand_embed_dict 的key
        })

    corpus_dataset = datasets.Dataset.from_list(corpus_rows)

    return query_dataset, corpus_dataset


DATASET_PARSER_NAME = "clotho_text_audio"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_clotho_text_audio_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "Clotho")
    path_info = EVAL_DATASET_LOCAL_PATH.get(dataset_name, EVAL_DATASET_HF_PATH.get(dataset_name))
    if path_info is None:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    return build_clotho_text_audio_dataset(path_info, **kwargs)
