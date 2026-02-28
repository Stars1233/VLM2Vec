"""
SpeechCOCO 文本->音频检索训练数据集处理（适配 train_collator_omni）。
- 数据源：本地 parquet 分片（MMEB-V3-train speechcoco）
- 训练样本形式：
    query_text = build_query_text("SpeechCOCO", text)
    pos_audio  = {"path": ..., "bytes": ..., "start": None, "end": None}
"""

import glob
import os
from typing import Any, Dict, List

import datasets
import pyarrow as pa

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.utils.basic_utils import print_master
from src.utils.dataset_utils import sample_dataset


POS_TEXT_AUDIO = "[AUDIO]"
DEFAULT_DATA_PATH = (
    "/data/mengrui/.cache/huggingface/datasets/MMEB-V3-train/audio-tasks-train/speechcoco"
)


def _resolve_parquet_files(data_path: str, data_subdir: str, parquet_pattern: str) -> List[str]:
    parquet_dir = os.path.join(data_path, data_subdir)
    files = sorted(glob.glob(os.path.join(parquet_dir, parquet_pattern)))
    if not files:
        raise FileNotFoundError(
            f"[SpeechCOCO] no parquet files found under {parquet_dir} with pattern={parquet_pattern}"
        )
    return files


def _load_parquet_dataset(parquet_files: List[str], cache_dir: str = None) -> datasets.Dataset:
    kwargs = {
        "path": "parquet",
        "data_files": {"train": parquet_files},
        "split": "train",
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        return datasets.load_dataset(**kwargs)
    except PermissionError:
        fallback_cache = os.path.join(os.path.dirname(parquet_files[0]), ".hf_datasets_cache")
        os.makedirs(fallback_cache, exist_ok=True)
        print_master(f"[SpeechCOCO] fallback cache_dir={fallback_cache}")
        kwargs["cache_dir"] = fallback_cache
        return datasets.load_dataset(**kwargs)


def _is_valid_row(row: Dict[str, Any]) -> bool:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return False
    audio = row.get("audio")
    if not isinstance(audio, dict):
        return False
    audio_path = audio.get("path")
    audio_bytes = audio.get("bytes")
    if audio_bytes is not None:
        if isinstance(audio_bytes, (bytes, bytearray)):
            return len(audio_bytes) > 0
        return True
    return isinstance(audio_path, str) and len(audio_path.strip()) > 0


def _build_audio_with_start_end(dataset: datasets.Dataset, column: str) -> datasets.Dataset:
    """
    将 struct<bytes,path> 规范成 struct<path,bytes,start,end>，避免 mixed_dataset.cast_column 失败。
    这里直接在 Arrow 层重建 struct 字段，复用原有 bytes/path 缓冲区，不复制大音频 payload。
    """
    if column not in dataset.column_names:
        raise KeyError(f"[SpeechCOCO] missing column: {column}")

    ds_table = dataset.data
    pa_table = ds_table.table if hasattr(ds_table, "table") else ds_table
    col_idx = pa_table.column_names.index(column)
    src_col = pa_table.column(column)

    out_chunks = []
    for chunk in src_col.chunks:
        if not pa.types.is_struct(chunk.type):
            raise TypeError(f"[SpeechCOCO] {column} must be a struct column, got={chunk.type}")
        field_names = set(chunk.type.names)

        path_arr = chunk.field("path") if "path" in field_names else pa.nulls(len(chunk), type=pa.string())
        bytes_arr = (
            chunk.field("bytes") if "bytes" in field_names else pa.nulls(len(chunk), type=pa.binary())
        )
        start_arr = (
            chunk.field("start")
            if "start" in field_names
            else pa.nulls(len(chunk), type=pa.float32())
        )
        end_arr = (
            chunk.field("end")
            if "end" in field_names
            else pa.nulls(len(chunk), type=pa.float32())
        )

        out_chunks.append(
            pa.StructArray.from_arrays(
                [path_arr, bytes_arr, start_arr, end_arr],
                names=["path", "bytes", "start", "end"],
            )
        )

    out_col = pa.chunked_array(out_chunks)
    out_table = pa_table.set_column(col_idx, column, out_col)
    return datasets.Dataset(out_table)


@AutoPairDataset.register("load_audio_speechcoco_dataset")
def load_audio_speechcoco_dataset(*args: Any, **kwargs: Any):
    """
    加载 SpeechCOCO 训练数据（文本->音频检索），适配 train_collator_omni。

    参数（kwargs）：
        data_path: 数据根目录（默认 MMEB-V3-train speechcoco 本地路径）
        data_subdir: parquet 子目录，默认 "data"
        parquet_pattern: 分片匹配模式，默认 "train-*.parquet"
        cache_dir: 可选，HF datasets cache 目录

    返回：
        datasets.Dataset，字段：
            - query_text/query_image/query_audio
            - pos_text/pos_image/pos_audio
            - dataset_infos
    """
    path_info = kwargs.get("path_info")
    if path_info is not None:
        data_path = path_info[0]
    else:
        data_path = kwargs.get("data_path", DEFAULT_DATA_PATH)

    data_subdir = kwargs.get("data_subdir", "data")
    parquet_pattern = kwargs.get("parquet_pattern", "train-*.parquet")
    cache_dir = kwargs.get("cache_dir", None)

    parquet_files = _resolve_parquet_files(data_path, data_subdir, parquet_pattern)
    print_master(f"[SpeechCOCO] loading parquet shards: {len(parquet_files)} files")
    dataset = _load_parquet_dataset(parquet_files, cache_dir=cache_dir)
    print_master(f"[SpeechCOCO] raw rows: {len(dataset)}")

    keep_cols = [
        c
        for c in [
            "id",
            "image_id",
            "audio",
            "text",
            "duration",
            "speaker",
            "gender",
            "nationality",
            "speed",
            "disfluency_pos",
            "disfluency_val",
        ]
        if c in dataset.column_names
    ]
    dataset = dataset.select_columns(keep_cols)
    dataset = sample_dataset(dataset, **kwargs)
    dataset = dataset.filter(_is_valid_row)

    if len(dataset) == 0:
        raise ValueError("[SpeechCOCO] no valid rows after filtering")

    dataset = dataset.rename_column("audio", "pos_audio")
    dataset = _build_audio_with_start_end(dataset, "pos_audio")

    texts = dataset["text"]
    query_texts: List[List[str]] = []
    for t in texts:
        q = build_query_text("SpeechCOCO", t)
        assert (
            isinstance(q, list)
            and len(q) == 1
            and isinstance(q[0], str)
            and q[0].strip()
        )
        query_texts.append(q)

    num_rows = len(dataset)
    ids = dataset["id"] if "id" in dataset.column_names else [None] * num_rows
    image_ids = dataset["image_id"] if "image_id" in dataset.column_names else [None] * num_rows
    durations = dataset["duration"] if "duration" in dataset.column_names else [None] * num_rows
    speakers = dataset["speaker"] if "speaker" in dataset.column_names else [None] * num_rows
    genders = dataset["gender"] if "gender" in dataset.column_names else [None] * num_rows
    nationalities = dataset["nationality"] if "nationality" in dataset.column_names else [None] * num_rows
    speeds = dataset["speed"] if "speed" in dataset.column_names else [None] * num_rows
    disfluency_pos = (
        dataset["disfluency_pos"] if "disfluency_pos" in dataset.column_names else [None] * num_rows
    )
    disfluency_val = (
        dataset["disfluency_val"] if "disfluency_val" in dataset.column_names else [None] * num_rows
    )

    dataset_infos: List[Dict[str, Any]] = []
    for i in range(num_rows):
        dataset_infos.append(
            {
                "id": ids[i],
                "image_id": image_ids[i],
                "duration": durations[i],
                "speaker": speakers[i],
                "gender": genders[i],
                "nationality": nationalities[i],
                "speed": speeds[i],
                "disfluency_pos": disfluency_pos[i],
                "disfluency_val": disfluency_val[i],
            }
        )

    dataset = dataset.add_column("query_text", query_texts)
    dataset = dataset.add_column("query_image", [None] * num_rows)
    dataset = dataset.add_column("query_audio", [None] * num_rows)
    dataset = dataset.add_column("pos_text", [POS_TEXT_AUDIO] * num_rows)
    dataset = dataset.add_column("pos_image", [None] * num_rows)
    dataset = dataset.add_column("dataset_infos", dataset_infos)

    dataset = dataset.select_columns(
        [
            "query_text",
            "query_image",
            "query_audio",
            "pos_text",
            "pos_image",
            "pos_audio",
            "dataset_infos",
        ]
    )

    try:
        setattr(dataset, "num_rows", len(dataset))
    except (AttributeError, TypeError):
        pass

    print_master(f"[SpeechCOCO] final rows: {len(dataset)}")
    return dataset
