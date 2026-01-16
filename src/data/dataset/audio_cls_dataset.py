"""
音频分类任务（训练/评测共用）的数据处理工具函数。
- 这里集中各数据集的加载与标签池构造，评测路由在 `eval_dataset/audio_cls_eval_dataset.py`。
"""

import glob
import os
from typing import Any, Dict, List, Tuple

import datasets
import numpy as np
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text


# -------- 通用工具 --------
def _get_label_fields(batch: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    选择一个可用的类别字段名，并返回该字段的列表。
    优先整数标签，再退回字符串标签。
    """
    int_cands = ["label", "target", "classID", "class_id", "instrument_family", "fold", "emotion_id"]
    str_cands = [
        "category",
        "label_name",
        "class",
        "classname",
        "instrument_family_str",
        "emotion",
        "major_emotion",
    ]
    for name in int_cands:
        if name in batch:
            return name, batch[name]
    for name in str_cands:
        if name in batch:
            return name, batch[name]
    raise ValueError(f"No label field found in batch keys={list(batch.keys())}")


def _extract_audio_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    将音频统一为 {path?, bytes?} 形式，供 collator 读取。
    支持HuggingFace数据集的array格式。
    """
    if "audio" in row and isinstance(row["audio"], dict):
        audio = row["audio"]
        audio_path = audio.get("path")
        audio_bytes = audio.get("bytes")

        # 新增：处理array格式的音频数据
        if audio_bytes is None and "array" in audio:
            import io
            import torch
            import torchaudio

            array = audio["array"]
            sampling_rate = audio.get("sampling_rate", 16000)

            # 如果是list，转换为numpy array
            if isinstance(array, list):
                array = np.array(array)

            # 转换为torch张量
            if array.ndim == 1:
                waveform = torch.from_numpy(array).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(array).float()

            # 保存到字节流
            buffer = io.BytesIO()
            torchaudio.save(buffer, waveform, sampling_rate, format='wav')
            audio_bytes = buffer.getvalue()

    else:
        audio_path = row.get("audio_path") or row.get("path") or None
        audio_bytes = row.get("audio_bytes") or None

    return {"path": audio_path, "bytes": audio_bytes}


def data_prepare(batch_dict, **kwargs):
    """
    输出字段：
    - query_audio: list[dict]  {path?, bytes?}
    - query_audio_path: list[str|None] 音频路径占位（若无则 None）
    - query_text: 占位空串
    - query_image: None 占位
    - cand_text: 全部类别名称（同一个 batch 复用）
    - cand_image: None 占位符列表（与 cand_text 长度相同）
    - dataset_infos: {label_id, label_name, cand_names}
    """
    label_field, labels = _get_label_fields(batch_dict)
    if kwargs.get("label_field_override") and kwargs["label_field_override"] in batch_dict:
        label_field = kwargs["label_field_override"]
        labels = batch_dict[label_field]

    # label 名称：如果字符串字段存在，优先用字符串；否则用整数->字符串
    label_names = None
    if kwargs.get("label_name_field_override") and kwargs["label_name_field_override"] in batch_dict:
        label_names = batch_dict[kwargs["label_name_field_override"]]
    if label_names is None:
        for key in ["category", "label_name", "class", "classname", "instrument_family_str", "emotion", "major_emotion"]:
            if key in batch_dict:
                label_names = batch_dict[key]
                break
    if label_names is None:
        label_names = [str(x) for x in labels]

    # 构建全集类别表
    all_label_names = kwargs["all_label_names"]
    label2id = {name: idx for idx, name in enumerate(all_label_names)}

    query_audio, query_audio_paths, query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], [], [], []
    for lbl, lbl_name, row_idx in zip(labels, label_names, range(len(labels))):
        row_dict = {k: v[row_idx] for k, v in batch_dict.items()}
        audio_obj = _extract_audio_obj(row_dict)
        query_audio.append(audio_obj)
        q_path = audio_obj.get("path") or row_dict.get("path") or row_dict.get("audio_path") or row_dict.get("filename")
        query_audio_paths.append(q_path)
        query_text = build_query_text(kwargs["dataset_name"])
        assert isinstance(query_text, list) and len(query_text) == 1 and isinstance(query_text[0], str) and query_text[0].strip()
        query_texts.append(query_text)
        query_images.append([None])
        cand_texts.append(all_label_names)
        cand_images.append([None] * len(all_label_names))
        lid = label2id.get(lbl_name, int(lbl) if isinstance(lbl, int) else 0)
        dataset_infos.append({"label_id": lid, "label_name": lbl_name, "cand_names": all_label_names})

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "query_audio_path": query_audio_paths,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos,
    }


# -------- NSynth --------
def _load_nsynth_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    # 对于 -1k 路径，数据在 eval/ 目录下；否则在 data/ 目录下
    if "-1k" in dataset_path:
        data_dir = os.path.join(dataset_path, "eval")
        query_file = os.path.join(data_dir, "query.parquet")
        if os.path.exists(query_file):
            parquet_files = [query_file]
        else:
            raise FileNotFoundError(f"query.parquet not found under {data_dir}")
    else:
        data_dir = os.path.join(dataset_path, "data")
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"NSynth parquet files not found under {data_dir}")

    split_name = split or "train"
    ds_dict = datasets.load_dataset("parquet", data_files={split_name: parquet_files})
    return ds_dict[split_name]


def _build_nsynth_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    int_candidates = [
        ("label_id", "label"),  # 新增：优先匹配整数ID字段
        ("instrument_family", "instrument_family_str"),
        ("instrument_family_id", "instrument_family_str"),
        ("label", "label_name"),
        ("target", "label_name"),
        ("class_id", "class"),
        ("classID", "class"),
    ]
    str_candidates = ["instrument_family_str", "label_name", "category", "class"]

    label_id_field = None
    label_name_field = None
    for int_field, str_field in int_candidates:
        if int_field in dataset.column_names:
            label_id_field = int_field
            if str_field in dataset.column_names:
                label_name_field = str_field
            break

    if label_id_field:
        label_ids = [int(x) for x in dataset[label_id_field]]
        if label_name_field:
            label_names = dataset[label_name_field]
        else:
            label_names = [str(x) for x in label_ids]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            lid = int(lid)
            id_to_name.setdefault(lid, str(lname))
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, label_id_field, label_name_field

    for str_field in str_candidates:
        if str_field in dataset.column_names:
            label_names = dataset[str_field]
            label_name_field = str_field
            break
    else:
        raise ValueError(f"NSynth: no usable label field in {dataset.column_names}")

    all_label_names = sorted(list(set(label_names)))
    return all_label_names, None, label_name_field


# -------- ESC-50 --------
def _load_esc50_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"ESC-50 parquet files not found under {dataset_path}")
    split_name = split or "train"
    ds_dict = datasets.load_dataset("parquet", data_files={split_name: parquet_files})
    return ds_dict[split_name]


def _build_esc50_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    targets = dataset["target"]
    categories = dataset["category"]
    id_to_name = {}
    for tid, cat in zip(targets, categories):
        tid = int(tid)
        id_to_name.setdefault(tid, str(cat))
    max_id = max(id_to_name.keys())
    all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
    return all_label_names, "target", "category"


# -------- UrbanSound8K --------
def _load_urbansound8k_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    csv_path = os.path.join(dataset_path, "csv_files", "test.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"UrbanSound8K csv not found: {csv_path}")
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError(f"UrbanSound8K csv missing path column, columns={df.columns}")
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(lambda x: {"path": os.path.normpath(os.path.join(dataset_path, x["path"]))})
    return dataset


def _build_urbansound8k_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    name_field = None
    for key in ["class", "classname"]:
        if key in dataset.column_names:
            name_field = key
            break
    if name_field is None:
        raise ValueError(f"UrbanSound8K: no label name field in {dataset.column_names}")

    id_field = None
    for key in ["classID", "label", "target"]:
        if key in dataset.column_names:
            id_field = key
            break

    if id_field:
        label_ids = [int(x) for x in dataset[id_field]]
        label_names = dataset[name_field]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            lid = int(lid)
            id_to_name.setdefault(lid, str(lname))
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, id_field, name_field

    label_names = dataset[name_field]
    all_label_names = sorted(list(set(label_names)))
    return all_label_names, None, name_field


# -------- CREMA-D --------
def _load_cremad_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    data_dir = os.path.join(dataset_path, "data")
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"CREMA-D parquet files not found under {data_dir}")
    split_name = split or "train"
    ds_dict = datasets.load_dataset("parquet", data_files={split_name: parquet_files})
    return ds_dict[split_name]


def _build_cremad_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    int_candidates = ["emotion_id", "label", "target"]
    str_candidates = ["emotion", "major_emotion"]

    label_id_field = None
    label_name_field = None

    for int_field in int_candidates:
        if int_field in dataset.column_names:
            label_id_field = int_field
            break

    for str_field in str_candidates:
        if str_field in dataset.column_names:
            label_name_field = str_field
            break

    if label_id_field:
        label_ids = [int(x) for x in dataset[label_id_field]]
        if label_name_field:
            label_names = dataset[label_name_field]
        else:
            label_names = [str(x) for x in label_ids]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            lid = int(lid)
            id_to_name.setdefault(lid, str(lname))
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, label_id_field, label_name_field

    if not label_name_field:
        raise ValueError(f"CREMA-D: no usable label field in {dataset.column_names}")
    label_names = dataset[label_name_field]
    all_label_names = sorted(list(set(label_names)))
    return all_label_names, None, label_name_field


# -------- SpeechCommands --------
def _load_speechcommand_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    """
    SpeechCommands:
    - 普通路径：用官方 split 列表 (testing_list.txt / validation_list.txt)，否则回退全量 wav
    - -1k 路径：直接加载 parquet（query_eval / query_train）
    """
    dataset_path, subset, split = path_info

    # -------------------------
    # 1) -1k parquet path
    # -------------------------
    if "-1k" in dataset_path:
        split_lower = (split or "eval").lower()
        # 你的 -1k 文件命名：query_eval.parquet / query_train.parquet
        if split_lower in ["test", "eval", "evaluation"]:
            query_file = "query_eval.parquet"
            split_name = "test"
        else:
            query_file = "query_train.parquet"
            split_name = "train"

        parquet_path = os.path.join(dataset_path, query_file)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"SpeechCommands parquet not found: {parquet_path}")

        ds_dict = datasets.load_dataset("parquet", data_files={split_name: [parquet_path]})
        ds = ds_dict[split_name]

        # 关键：兼容字段（你的 parquet 里 label 是字符串类别名，label_id 是整数）
        # 如果缺 label_id，就从 label 做一个稳定映射（尽量别走到这）
        if "label_id" not in ds.column_names:
            if "label" not in ds.column_names:
                raise ValueError(f"SpeechCommands -1k: missing label/label_id, columns={ds.column_names}")
            uniq = sorted(set(str(x) for x in ds["label"]))
            name2id = {n: i for i, n in enumerate(uniq)}
            ds = ds.map(lambda x: {"label_id": int(name2id[str(x["label"])])})

        # 如果缺 label_name，用 label 顶上（label_name 仅用于显示/构建 pool）
        if "label_name" not in ds.column_names and "label" in ds.column_names:
            ds = ds.map(lambda x: {"label_name": str(x["label"])})
        # 如果两者都没有，直接报错
        if "label_name" not in ds.column_names:
            raise ValueError(f"SpeechCommands -1k: missing label_name/label, columns={ds.column_names}")

        # 统一：确保有 audio / audio_path（你现在 parquet 里两者都有）
        # 如果只有 audio_path，就让后续 _extract_audio_obj 能读到 path
        if "audio_path" not in ds.column_names and "path" in ds.column_names:
            ds = ds.rename_column("path", "audio_path")

        return ds

    # -------------------------
    # 2) original wav loading
    # -------------------------
    split_lower = (split or "").lower()
    list_file = None
    if split_lower == "test":
        list_file = "testing_list.txt"
    elif split_lower in ["val", "validation"]:
        list_file = "validation_list.txt"

    rel_paths: List[str] = []
    if list_file:
        list_path = os.path.join(dataset_path, list_file)
        if os.path.exists(list_path):
            with open(list_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rel_paths.append(line)

    if not rel_paths:
        # 回退：遍历目录收集 wav（排除前缀 "_" 的背景噪声）
        for dirpath, _, filenames in os.walk(dataset_path):
            base = os.path.basename(dirpath)
            if base.startswith("_"):
                continue
            for fname in filenames:
                if fname.lower().endswith(".wav"):
                    abs_path = os.path.join(dirpath, fname)
                    rel_paths.append(os.path.relpath(abs_path, dataset_path))

    label_names: List[str] = []
    abs_paths: List[str] = []
    for rel in rel_paths:
        parts = rel.split("/")
        if not parts:
            continue
        label = parts[0]
        if label.startswith("_"):
            continue
        abs_paths.append(os.path.normpath(os.path.join(dataset_path, rel)))
        label_names.append(label)

    unique_labels = sorted(set(label_names))
    label2id = {n: i for i, n in enumerate(unique_labels)}
    label_ids = [label2id[l] for l in label_names]

    dataset = datasets.Dataset.from_dict(
        {
            "path": abs_paths,
            "audio_path": abs_paths,      # 兼容你后续用 audio_path 的逻辑
            "label_id": label_ids,        # 统一用 label_id 做 gold
            "label_name": label_names,    # 字符串类别名
            "label": label_names,         # 也放一份，兼容旧逻辑（label 作为 name）
        }
    )
    return dataset


def _build_speechcommand_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    """
    目标：返回的 all_label_names 必须满足 all_label_names[i] 对应 label_id==i
    否则分类评测会“不报错但分数漂移”。

    返回:
      all_label_names, label_field_override, label_name_field_override
    """
    cols = dataset.column_names

    # 1) id 字段：优先 label_id，其次 label(若为 int)
    if "label_id" in cols:
        id_field = "label_id"
    elif "label" in cols:
        # 只有当 label 真的是整数标签才用它
        sample = dataset["label"][: min(50, len(dataset))]
        if all(isinstance(x, (int,)) for x in sample):
            id_field = "label"
        else:
            id_field = None
    else:
        id_field = None

    # 2) name 字段：优先 label_name，否则用 label（字符串类别名）
    if "label_name" in cols:
        name_field = "label_name"
    elif "label" in cols:
        name_field = "label"
    else:
        raise ValueError(f"SpeechCommands: missing label name field, columns={cols}")

    # 3) 构造 pool：如果有 id_field，就按 id 对齐；没有就按名字排序（不推荐）
    if id_field is not None:
        label_ids = [int(x) for x in dataset[id_field]]
        label_names = [str(x) for x in dataset[name_field]]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            if lid not in id_to_name:
                id_to_name[lid] = lname
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, id_field, name_field

    # fallback（尽量不要走到这）
    all_label_names = sorted(list(set(str(x) for x in dataset[name_field])))
    return all_label_names, None, name_field


# -------- 主入口（供评测路由调用） --------
def build_audio_cls_dataset(dataset_name: str, path_info: Tuple[str, str, str], **kwargs):
    """
    构建音频分类检索式评测数据集，返回 (dataset, corpus)。
    路由逻辑在 eval_dataset/audio_cls_eval_dataset.py 调用。
    """
    if dataset_name == "NSynth":
        dataset = _load_nsynth_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_nsynth_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        if label_field_override:
            kwargs["label_field_override"] = label_field_override
        if label_name_field_override:
            kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "ESC-50":
        dataset = _load_esc50_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_esc50_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        kwargs["label_field_override"] = label_field_override
        kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "UrbanSound8K":
        dataset = _load_urbansound8k_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_urbansound8k_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        if label_field_override:
            kwargs["label_field_override"] = label_field_override
        if label_name_field_override:
            kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "CREMA-D":
        dataset = _load_cremad_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_cremad_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        if label_field_override:
            kwargs["label_field_override"] = label_field_override
        if label_name_field_override:
            kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "SpeechCommands":
        dataset = _load_speechcommand_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_speechcommand_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        kwargs["label_field_override"] = label_field_override
        kwargs["label_name_field_override"] = label_name_field_override
    else:
        dataset = load_hf_dataset(path_info)
        dataset = sample_dataset(dataset, **kwargs)

        label_field, labels = _get_label_fields(dataset)
        label_names = None
        for key in ["category", "label_name", "class", "instrument_family_str"]:
            if key in dataset.column_names:
                label_names = dataset[key]
                break
        if label_names is None:
            label_names = [str(x) for x in dataset[label_field]]
        all_label_names = sorted(list(set(label_names)))

        kwargs["all_label_names"] = all_label_names
        kwargs.pop("label_field_override", None)
        kwargs.pop("label_name_field_override", None)

    # 添加dataset_name到kwargs，供data_prepare使用
    kwargs["dataset_name"] = dataset_name

    dataset = sample_dataset(dataset, **kwargs)

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
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
            "query_audio_path",
            "cand_text",
            "cand_image",
            "dataset_infos",
        ]
    )
    corpus = None
    return dataset, corpus