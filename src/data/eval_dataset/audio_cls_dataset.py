"""
音频分类（检索式）评测路由：
- 仅负责路径解析与注册，具体数据处理在 src/data/dataset/audio_cls_dataset.py
"""

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.data.dataset.audio_cls_dataset import build_audio_cls_dataset


DATASET_PARSER_NAME = "audio_cls"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_audio_cls_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.pop("dataset_name")
    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
    else:
        path_info = EVAL_DATASET_HF_PATH[dataset_name]

    return build_audio_cls_dataset(dataset_name, path_info, **kwargs)

