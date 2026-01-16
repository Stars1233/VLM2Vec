"""
TUTSound 音频事件检测数据集处理
- 文本标签 -> 音频检索评测数据处理
"""

import os
from typing import Dict, List, Tuple, Any

import datasets
from src.utils.dataset_utils import sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH


def _read_evaluate_file(eval_path: str) -> List[Tuple[str, str, float, float, str]]:
    rows = []
    with open(eval_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = line.split()
            if len(parts) < 5:
                continue
            fp = parts[0].strip()
            scene = parts[1].strip()
            onset = float(parts[2])
            offset = float(parts[3])
            label = parts[4].strip()
            rows.append((fp, scene, onset, offset, label))
    return rows


def build_tutsound_audio_dataset(path_info: Tuple[str, str, str], **kwargs):
    dataset_path, subset, split = path_info
    base_dir = os.path.join(dataset_path, "TUT-sound-events-2017-development")

    # 1) 读四个 fold 的 evaluate 文件
    all_rows = []
    for fold in ["1", "2", "3", "4"]:
        eval_file_name = f"street_fold{fold}_evaluate.txt"
        eval_file = os.path.join(base_dir, "evaluation_setup", eval_file_name)
        if os.path.isfile(eval_file):
            fold_rows = _read_evaluate_file(eval_file)
            all_rows.extend(fold_rows)
            print(f"TUTSound: loaded fold{fold} with {len(fold_rows)} events")
        else:
            print(f"Warning: TUTSound evaluate file not found: {eval_file}")

    if not all_rows:
        raise FileNotFoundError("No TUTSound evaluate files found for any fold")

    # 2) 按 wav 聚合 GT 事件段
    gt_by_fp: Dict[str, Dict[str, Any]] = {}
    for fp, scene, onset, offset, label in all_rows:
        if fp not in gt_by_fp:
            gt_by_fp[fp] = {"scene": scene, "events": []}
        gt_by_fp[fp]["events"].append({"label": label, "onset": onset, "offset": offset})

    # 3) 构建 corpus：每个 event_label 选一个 exemplar wav（作为全局候选池）
    exemplar_by_label: Dict[str, str] = {}
    for fp in sorted(gt_by_fp.keys()):
        for ev in gt_by_fp[fp]["events"]:
            lab = ev["label"]
            if lab not in exemplar_by_label:
                exemplar_by_label[lab] = fp

    # 全局候选名列表（顺序=corpus行顺序）
    cand_names: List[str] = []
    cand_text_corpus: List[List[str]] = []
    cand_audio_corpus: List[List[Dict[str, Any]]] = []
    cand_image_corpus: List[List[Any]] = []
    infos_corpus: List[Dict[str, Any]] = []

    # 候选文本：必须非空（Omni_process_fn 会要求 text 非空）
    cand_text_item = build_query_text("TUTSound")
    assert (
        isinstance(cand_text_item, list)
        and len(cand_text_item) == 1
        and isinstance(cand_text_item[0], str)
        and cand_text_item[0].strip()
    )

    for lab in sorted(exemplar_by_label.keys()):
        fp = exemplar_by_label[lab]
        abs_path = os.path.normpath(os.path.join(base_dir, fp))

        cid = lab  # candidate id = label（稳定）
        cand_names.append(cid)

        # ✅ corpus 每行一个 candidate，所以 cand_* 都是长度=1 的 list
        cand_text_corpus.append(cand_text_item)  # List[str] len=1
        cand_audio_corpus.append([{"path": abs_path, "bytes": None}])
        cand_image_corpus.append([None])

        # ✅ corpus 行的 dataset_infos：每行 cand_names 必须 len=1
        infos_corpus.append(
            {
                "cand_names": [cid],
                "label_name": cid,
                "corpus_id": cid,
            }
        )

    corpus = datasets.Dataset.from_dict(
        {
            "cand_text": cand_text_corpus,
            "cand_audio": cand_audio_corpus,
            "cand_image": cand_image_corpus,
            "dataset_infos": infos_corpus,
        }
    )

    # 4) 构造 query：每个 wav 一条；label_name 是它包含的 event_labels（用于评测）
    query_audio = []
    query_text = []
    query_image = []
    cand_text = []
    cand_image = []
    cand_audio = []
    dataset_infos = []

    wav_list = sorted(gt_by_fp.keys())

    for fp in wav_list:
        abs_path = os.path.normpath(os.path.join(base_dir, fp))
        info = gt_by_fp[fp]

        gt_labels = sorted({ev["label"] for ev in info["events"]})

        query_audio.append({"path": abs_path, "bytes": None})
        query_text.append(cand_text_item)
        query_image.append([None])

        # ✅ query 不携带候选（global 模式用 corpus）
        cand_text.append([])
        cand_image.append([])
        cand_audio.append([])

        # ✅ 关键修改：不要在 query 的 dataset_infos 里放 cand_names
        dataset_infos.append(
            {
                "file_path": fp,
                "scene": info["scene"],
                "gt_events": info["events"],
                "label_name": gt_labels,   # ✅ eval.py 需要
                "label_cand_id": -1,
            }
        )

    dataset = datasets.Dataset.from_dict(
        {
            "query_text": query_text,
            "query_image": query_image,
            "query_audio": query_audio,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "cand_audio": cand_audio,
            "dataset_infos": dataset_infos,
        }
    )

    dataset = sample_dataset(dataset, **kwargs)
    return dataset, corpus


DATASET_PARSER_NAME = "tutsound_audio_gnd"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_tutsound_audio_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "TUTSound")
    path_info = EVAL_DATASET_LOCAL_PATH.get(dataset_name, EVAL_DATASET_HF_PATH.get(dataset_name))
    if path_info is None:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    return build_tutsound_audio_dataset(path_info, **kwargs)

