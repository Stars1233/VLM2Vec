from typing import List
from datasets import load_dataset
import os
import json

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils.basic_utils import print_master, print_rank


def create_empty_image_dict(image_resolution):
    """创建一个空的图像字典，用于纯文本数据"""
    return {
        "bytes": [None], 
        "paths": [None], 
        "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]
    }


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    """
    处理ToolRet数据集的训练数据
    数据格式：query_text, query_id, pos_text, pos_id
    这是一个纯文本数据集，不包含图像
    """
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']

    batch_size = len(batch_dict['qry_text'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    
    for qry_text, pos_text, neg_text in \
        zip(batch_dict['qry_text'], 
            batch_dict['pos_text'],
            batch_dict.get('neg_text', [''] * batch_size)):
        
        if not qry_text or not pos_text:
            print("empty inputs")
            continue
            
        # 处理不同模型的特殊token
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else ''
        
        query_texts.append(qry_text)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
        
        # 为纯文本数据创建空的图像占位符
        empty_image = create_empty_image_dict(image_resolution)
        query_images.append(empty_image)
        pos_images.append(empty_image)
        neg_images.append(empty_image)
    
    if len(query_texts) == 0:
        print('something went wrong')
    
    return {
        "query_text": query_texts, "query_image": query_images,
        "pos_text": pos_texts, "pos_image": pos_images,
        "neg_text": neg_texts, "neg_image": neg_images
    }


DATASET_PARSER_NAME = "toolDe"

@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_toolde_dataset(model_args, data_args, training_args, *args, **kwargs):
    """
    加载ToolDe测试数据集
    
    参数:
        dataset_name: 数据集名称，默认为"toolde"
        subset_name: 子任务名称，例如"apibank"
        dataset_split: 数据集分割，默认为"test"
        data_path: 数据文件路径（可选，如果使用本地文件）
    """
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "test")
    data_path = kwargs.get("data_path", None)
    
    # 如果提供了data_path，从本地加载
    if data_path and os.path.exists(data_path):
        print_master(f"Loading {dataset_name}/{subset_name} from local path: {data_path}")
        # 从JSONL文件加载数据
        dataset = load_dataset('json', data_files=data_path, split='test')
    else:
        # 从HuggingFace加载
        dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")
    
    column_names = dataset.column_names
    num_rows = dataset.num_rows
    
    # 限制样本数量
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    
    # 转换为可迭代数据集
    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)
    
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    
    # 移除原始列
    remove_columns = ['qry_text', 'pos_text', 'qry_id', 'pos_id']
    if 'neg_text' in column_names:
        remove_columns.append('neg_text')
        remove_columns.append('neg_id')
    
    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs), 
        batched=True, 
        batch_size=128,
        remove_columns=remove_columns,
        drop_last_batch=True
    )
    
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, 'num_rows', num_rows)
    print_master(f"Loaded {DATASET_PARSER_NAME}/{subset_name} dataset with {num_rows} samples")
    return dataset

