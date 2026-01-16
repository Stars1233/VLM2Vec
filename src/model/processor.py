import logging

import PIL
from transformers.image_utils import ChannelDimension

from src.model.baseline_backbone.colpali import ColPaliProcessor

logger = logging.getLogger(__name__)

import torch
import numpy as np
from src.utils.basic_utils import print_master

from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from src.model.vlm_backbone.qwen2_vl_tokenselection import \
    Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration, \
    Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor
from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.model.vlm_backbone.qwen2_5_vl_tokenselection import \
    Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration
from src.model.vlm_backbone.omni_embed import OmniEmbedForConditionalGeneration


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000
TEXT_ONLY_MAX_LEN = 2048


def _text_only_max_length(max_length):
    return TEXT_ONLY_MAX_LEN if max_length is None else max_length

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
QWEN2_5_OMNI = 'qwen2_5_omni'  # Qwen2.5-Omni / Omni-Embed
INTERNVIDEO2 = 'internvideo2'
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
LamRA_QWEN2_5 = 'lamra_qwen25'  # QWEN2.5-VL
COLPALI = 'colpali'  # PaliGemma-3B
E5_V = 'e5_v'  # Llava_next
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'qwen2_5_omni': QWEN2_5_OMNI,
    'nvomniembed': QWEN2_5_OMNI,
    'internvideo2': INTERNVIDEO2,
    'gme': GME, 
    'lamra': LamRA,
    'lamra_qwen25': LamRA,
    'colpali': COLPALI,
    'e5_v': E5_V,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_OMNI: "<|image_pad|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    LamRA_QWEN2_5: "<|image_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_OMNI: "<|video_pad|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    LamRA_QWEN2_5: "<|video_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    QWEN2_5_OMNI: OmniEmbedForConditionalGeneration,
    INTERNVIDEO2: InternVideo2_Stage2,
    E5_V: LlavaNextForConditionalGeneration,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.baseline_backbone.llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer, size=size
        )
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone in [QWEN2_5_VL, LamRA_QWEN2_5]:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)
    elif model_args.model_backbone == QWEN2_5_VL_TOKENSELECTION:
        # TODO: qwen2.5 token selection not working yet
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone == QWEN2_5_OMNI:
        # Qwen2.5-Omni / Omni-Embed：支持文本、图像、视频、音频的多模态处理器
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        from src.model.vlm_backbone.omni_embed.audio_processing_qwen2_5_omni import Qwen2_5_OmniAudioProcessor

        # 1. 图像/视频 resize 配置
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        
        # 2. 初始化图像处理器（同时支持图像和视频）
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        
        # 3. 初始化文本 tokenizer
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        
        # 4. 创建基础 processor（支持文本、图像、视频）
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path, 
            image_processor=image_processor, 
            tokenizer=tokenizer
        )
        
        # 5. 添加音频处理器（生成 128-dim 梅尔频谱图，与 Qwen2.5-Omni 模型兼容）
        processor.audio_processor = Qwen2_5_OmniAudioProcessor(
            sample_rate=16000,  # 音频采样率
            mono=True,          # 转换为单声道
            normalize=True,     # 归一化音频信号
            dtype=torch.float32,
            n_mels=128,         # 梅尔频谱图通道数（与模型期望一致）
            n_fft=400,          # FFT 窗口大小（25ms @ 16kHz）
            hop_length=160,     # 跳跃长度（10ms @ 16kHz）
            f_min=0.0,
            f_max=8000.0,
        )
        
    elif model_args.model_backbone == INTERNVIDEO2:
        return None
    elif model_args.model_backbone == COLPALI:
        from transformers import AutoProcessor
        processor = ColPaliProcessor.from_pretrained(model_args.model_name)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO: NOT FINISHED YET!
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        # in theory, each batch item should contain a list of frames, but we still check for exceptions here
        # if no images as input (not likely to happen in mmeb pro cases)
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(
                images=None,
                text=text,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
        else:
            image_exists = True
            # in theory, valid images should be a list of frames
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        # 'texts': texts,
        # 'images': visual_inputs,
    }
    image_exists = any([p is not None for p in pixel_values])
    if image_exists:
        pixel_values = torch.from_numpy(np.array(pixel_values)).float()
        pixel_values_shape = pixel_values.shape
        pixel_values = pixel_values.reshape(pixel_values_shape[0] * pixel_values_shape[1], *pixel_values_shape[2:])
        image_sizes = torch.tensor(np.array(image_sizes)).long()
        image_sizes_shape = image_sizes.shape
        image_sizes = image_sizes.reshape(image_sizes_shape[0] * image_sizes_shape[1], *image_sizes_shape[2:])
        inputs['pixel_values'] = torch.from_numpy(np.array(pixel_values)).float()
        inputs['image_sizes'] = torch.tensor(np.array(image_sizes)).long()
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(
                text,
                None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(
                text=[text],
                images=None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            images[iid] = image
                    inputs = processor(text=[text], images=images, return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                for i in images:
                    print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs

def Gme_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    inputs = {
        'texts': model_inputs['text'],
        'images': model_inputs['images'],
    }
    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(
                text=[text],
                images=None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)
            select_mask.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    if image_exists:
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else (torch.zeros(patch_pos_shape_for_padding) - 1) for v in patch_pos]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)

    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos
    inputs['select_mask'] = select_mask

    return inputs


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO not working yet
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(
                text,
                None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def ColPali_process_fn(model_inputs: dict, processor, max_length=None):
    texts, images = model_inputs['text'], model_inputs['images']
    
    input_ids_batch = []
    attention_mask_batch = []
    pixel_values_batch = []

    for text, image in zip(texts, images):
        if image is not None:
            inputs = processor.process_images([image])
            pixel_values_batch.append(inputs['pixel_values'])
        else:
            inputs = processor.process_queries([text], max_length=_text_only_max_length(max_length))
            pixel_values_batch.append(None)

        input_ids_batch.append(inputs['input_ids'].squeeze().tolist())
        attention_mask_batch.append(inputs['attention_mask'].squeeze().tolist())

    # Pad input_ids and attention_mask
    padded_text_inputs = processor.tokenizer.pad(
        {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch},
        return_tensors="pt"
    )
    
    final_input_ids = padded_text_inputs['input_ids']
    final_attention_mask = padded_text_inputs['attention_mask']

    # Handle pixel_values
    if any(pv is not None for pv in pixel_values_batch):
        # Find a representative shape for pixel_values
        representative_pv_shape = None
        for pv in pixel_values_batch:
            if pv is not None:
                representative_pv_shape = pv.shape
                break
        
        processed_pixel_values = []
        for pv in pixel_values_batch:
            if pv is None:
                # Create a zero tensor of the representative shape
                processed_pixel_values.append(torch.zeros(representative_pv_shape))
            else:
                processed_pixel_values.append(pv)
        final_pixel_values = torch.cat(processed_pixel_values)
    else:
        # No images in the batch at all
        batch_size = len(texts)
        # SigLIP expects 3 channels (RGB) and a square image of 448x448 based on the error (1024 patches = 32x32 patches, with patch_size=14, 32*14=448)
        default_channels = 3
        default_height = 448
        default_width = 448
        final_pixel_values = torch.zeros(batch_size, default_channels, default_height, default_width)

    return {
        'input_ids': final_input_ids,
        'attention_mask': final_attention_mask,
        'pixel_values': final_pixel_values,
    }


def InternVideo2_process_fn(model_inputs: dict, processor, max_length=None):
    if all(x is None for x in model_inputs["images"]):
        # Text side
        from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        inputs = tokenizer(
            model_inputs["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt")
    else:
        # Video side
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert from PIL image to tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # ImageNet std
        ])
        frame_list = model_inputs["images"]
        # to make image inputs be exact 4 frames
        # Case 1: frame_list is flat (not a list of lists), e.g., [PIL, PIL, ...]
        if type(frame_list[0]) is not list:
            frame_list = [[img.copy() for _ in range(4)] for img in frame_list]
        # Case 2: frame_list is already a list of lists, ensure each has exactly 4 images
        elif type(frame_list[0]) is list and len(frame_list[0]) != 4:
            new_list = []
            for frames in frame_list:
                if len(frames) < 4:
                    frames = frames + [frames[-1].copy() for _ in range(4 - len(frames))]
                elif len(frames) > 4:
                    # Sample 4 indices uniformly across the sequence
                    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
                    frames = [frames[i] for i in indices]
                new_list.append(frames)
            frame_list = new_list
        pixel_values = [
            torch.stack([preprocess(img) for img in frames], dim=0)  # (num_frames, C, H, W)
            for frames in frame_list
        ]

        pixel_values = torch.stack(pixel_values, dim=0)  # (B, num_frames, C, H, W)
        inputs = {'pixel_values': pixel_values}

    return inputs


def Omni_process_fn(model_inputs: dict, processor, max_length=None):
    """
    Qwen2.5-Omni / Omni-Embed processor for EVAL (collate-time).
    Safe rules:
    - Always returns torch tensors for input_ids / attention_mask (LongTensor).
    - Supports text-only / text+image / text+video / text+audio (audio optional).
    - Does NOT rely on Dataset.map storing tensors (avoid Arrow -> list issue).
    """
    import inspect
    import torch
    import PIL

    texts = model_inputs.get("text", [])
    visual_inputs = model_inputs.get("images", None)  # could be None or list
    if visual_inputs is None and "videos" in model_inputs:
        # AVE/cand_video uses videos; Omni path relies on text token to pick video handling.
        visual_inputs = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)         # could be None or list
    audio_sample_rate = model_inputs.get("audio_sample_rate", None)

    if texts is None:
        texts = []
    if visual_inputs is None:
        visual_inputs = [None] * len(texts)
    if audios is None:
        audios = [None] * len(texts)

    assert len(texts) == len(visual_inputs), f"len(texts)={len(texts)} != len(images)={len(visual_inputs)}"
    assert len(texts) == len(audios), f"len(texts)={len(texts)} != len(audios)={len(audios)}"

    if not any(t and str(t).strip() for t in texts):
        raise ValueError("Omni_process_fn: at least one non-empty text is required.")

    vlm_image_token = VLM_IMAGE_TOKENS[QWEN2_5_OMNI]
    vlm_video_token = VLM_VIDEO_TOKENS[QWEN2_5_OMNI]

    # -----------------------------------------
    # 1) Audio: batch process (optional)
    # -----------------------------------------
    input_features = None
    audio_attention_mask = None
    audio_feature_lengths = None

    if hasattr(processor, "audio_processor") and any(a is not None for a in audios):
        # collect valid waveforms
        audio_batch = []
        audio_pos = []  # mapping from packed idx -> original idx

        for i, a in enumerate(audios):
            if a is None:
                continue
            if not isinstance(a, torch.Tensor) or a.ndim != 1:
                raise ValueError(f"Audio must be 1D torch.Tensor waveform, got {type(a)} shape={getattr(a,'shape',None)}")
            audio_batch.append(a.detach().cpu().numpy())
            audio_pos.append(i)

        sig = inspect.signature(processor.audio_processor.__call__)
        kwargs = {"return_tensors": "pt"}
        if "sampling_rate" in sig.parameters and audio_sample_rate is not None:
            kwargs["sampling_rate"] = audio_sample_rate

        audio_out = processor.audio_processor(audio_batch, **kwargs)

        feats = audio_out["input_features"]  # (B_valid, 128, T)
        fam = audio_out.get("feature_attention_mask", None)
        afl = audio_out.get("audio_feature_lengths", None)

        # re-expand to full batch length (keep None for missing)
        input_features = [None] * len(audios)
        audio_attention_mask = [None] * len(audios)
        audio_feature_lengths = [None] * len(audios)

        for j, orig_i in enumerate(audio_pos):
            input_features[orig_i] = feats[j]
            audio_attention_mask[orig_i] = (fam[j] if fam is not None else None)
            audio_feature_lengths[orig_i] = (afl[j] if afl is not None else None)

    # -----------------------------------------
    # 2) Text + Vision: per-sample tokenize
    # -----------------------------------------
    input_ids_list = []
    pixel_values_list = []
    image_grid_thw_list = []
    pixel_values_videos_list = []
    video_grid_thw_list = []

    for text, vis in zip(texts, visual_inputs):
        text = str(text)

        # -------- text only --------
        if vis is None or (isinstance(vis, list) and any(v is None for v in vis)):
            out = processor(
                text=[text],
                return_tensors="pt",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_ids_list.append(out["input_ids"][0].tolist())
            pixel_values_list.append(None)
            image_grid_thw_list.append(None)
            pixel_values_videos_list.append(None)
            video_grid_thw_list.append(None)
            continue
        
        
        def _squeeze_leading_ones(x, max_squeeze=2):
            if x is None:
                return None
            if not isinstance(x, torch.Tensor):
                return x
            for _ in range(max_squeeze):
                if x.dim() >= 1 and x.size(0) == 1:
                    x = x.squeeze(0)
                else:
                    break
            return x

        # -------- image token path --------
        if vlm_image_token in text:
            # allow vis as single PIL or list[PIL]
            if isinstance(vis, PIL.Image.Image):
                vis = [vis]

            out = processor(
                text=[text],
                images=vis,
                return_tensors="pt",
                truncation=False,
            )

            input_ids_list.append(out["input_ids"][0].tolist())

            pv = _squeeze_leading_ones(out.get("pixel_values", None), max_squeeze=2)
            gt = _squeeze_leading_ones(out.get("image_grid_thw", None), max_squeeze=2)
            pixel_values_list.append(pv)        # ✅ 用 squeeze 后的
            image_grid_thw_list.append(gt)      # ✅ 用 squeeze 后的

            pixel_values_videos_list.append(None)
            video_grid_thw_list.append(None)
            continue

        # -------- video token path --------
        if vlm_video_token in text:
            # vis MUST be list[PIL.Image] (frames)
            if not isinstance(vis, list) or (len(vis) > 0 and not isinstance(vis[0], PIL.Image.Image)):
                raise ValueError(f"Video input must be List[PIL.Image] frames, got {type(vis)} elem={type(vis[0]) if isinstance(vis,list) and len(vis)>0 else None}")
            out = processor(
                text=[text],
                videos=[vis],
                return_tensors="pt",
                truncation=False,
            )
            input_ids_list.append(out["input_ids"][0].tolist())
            pixel_values_list.append(None)
            image_grid_thw_list.append(None)
            pixel_values_videos_list.append(out.get("pixel_values_videos", None))
            video_grid_thw_list.append(out.get("video_grid_thw", None))
            continue

        # If you provide visual input but text has no token -> explicit error helps debugging
        raise ValueError(f"Visual input is provided but text contains no visual token ({vlm_image_token} / {vlm_video_token}). text={text[:200]}")

    # -----------------------------------------
    # 3) Pad text (ALWAYS tensorize here)
    # -----------------------------------------
    enc = processor.tokenizer.pad(
        {"input_ids": input_ids_list},
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"].long()
    attention_mask = enc["attention_mask"].long()

    # -----------------------------------------
    # 4) Clean empty vision lists (keep None if whole batch has no that modality)
    # -----------------------------------------
    if not any(v is not None for v in pixel_values_list):
        pixel_values_list = None
        image_grid_thw_list = None
    if not any(v is not None for v in pixel_values_videos_list):
        pixel_values_videos_list = None
        video_grid_thw_list = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values_list,
        "image_grid_thw": image_grid_thw_list,
        "pixel_values_videos": pixel_values_videos_list,
        "video_grid_thw": video_grid_thw_list,
        "input_features": input_features,                 # list[Tensor|None] or None
        "audio_attention_mask": audio_attention_mask,     # list[Tensor|None] or None
        "audio_feature_lengths": audio_feature_lengths,   # list[Tensor|None] or None
    }


def e5_v_prompt_template(text, add_video_token, add_image_token):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    if text is not None and add_video_token is False and add_image_token is False:  # only text
        prompt = llama3_template.format('{}\nSummary above sentence in one word: '.format(text))
    if text is None and add_video_token:  # only video
        prompt = llama3_template.format('<image>\nSummary above video in one word: ')
    if text is None and add_image_token:  # only image
        prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    if text is not None and add_video_token:  # video + text
        prompt = llama3_template.format('<image>\n{}\nSummary above video and text in one word: '.format(text))
    if text is not None and add_image_token:
        prompt = llama3_template.format('<image>\n{}\nSummary above image and text in one word: '.format(text))

    return prompt


PROMPT_TEMPLATE_DICT = {
    "e5_v": e5_v_prompt_template,
}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    # TBD: Reorganize the hard-code part for baselines such as internvideo2
    if model_backbone == "internvideo2":
        return text
    elif model_backbone in [GME, LamRA, LamRA_QWEN2_5]:
        if text:
            return instruction + " " + text # GME and LamRA do not need special tokens
        else:
            return instruction + " "
    elif model_backbone == E5_V:
        return PROMPT_TEMPLATE_DICT[model_backbone](text, add_video_token, add_image_token)

    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt


process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_OMNI: Omni_process_fn,
    INTERNVIDEO2: InternVideo2_process_fn,
    GME: Gme_process_fn,
    LamRA: Gme_process_fn,
    LamRA_QWEN2_5: Gme_process_fn,
    COLPALI: ColPali_process_fn,
    E5_V: Llava_NEXT_process_fn,
}
