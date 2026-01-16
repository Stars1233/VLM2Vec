"""
AVE 音频->视频检索数据集工具（MSRVTT风格）：
- 解析 split 文件 (testSet.txt)
- query: audio
- candidates: video represented by N frames (List[PIL.Image])
"""

import os
from typing import List, Dict, Any, Tuple

import datasets
from PIL import Image

from src.model.processor import process_input_text
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames


TASK_INST_TGT = "Understand the content of the provided video."


def parse_ave_split(split_file: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Category"):
                continue
            parts = line.split("&")
            if len(parts) < 5:
                continue
            category, video_id, quality, start, end = parts[:5]
            clip_id = f"{video_id}_{int(float(start))}_{int(float(end))}"
            samples.append(
                {
                    "category": category,
                    "video_id": video_id,
                    "quality": quality,
                    "start": float(start),
                    "end": float(end),
                    "clip_id": clip_id,
                }
            )
    return samples


def load_audio_ave_dataset(path_info: Tuple[str, str, str], **kwargs):
    """
    返回 (dataset, corpus=None)

    query:
      - query_text: List[str]（只放一个指令）
      - query_audio: [{"path": ..., "bytes": None}]

    candidates (global fixed pool):
      - cand_video: List[List[PIL.Image]]  (num_frames=8)
      - cand_text:  List[str] (带 video token 的指令)
    """
    data_path = path_info[0]
    split_file: str = kwargs.get("split_file", "testSet.txt")
    audio_dir: str = kwargs.get("audio_dir", "audios")
    video_dir: str = kwargs.get("video_dir", "AVE")

    frame_root: str = kwargs.get("frame_root", os.path.join(data_path, "frames"))
    num_frames: int = kwargs.get("num_frames", 8)
    max_frames_saved: int = kwargs.get("max_frames_saved", 100)

    model_backbone = kwargs.get("model_backbone", None)
    if model_backbone is None:
        raise ValueError("[AVE] model_backbone is required")

    split_path = os.path.join(data_path, split_file)
    audio_root = os.path.join(data_path, audio_dir)
    video_root = os.path.join(data_path, video_dir)

    assert os.path.isfile(split_path), f"未找到 split 文件: {split_path}"
    assert os.path.isdir(audio_root), f"未找到音频目录: {audio_root}"
    assert os.path.isdir(video_root), f"未找到视频目录: {video_root}"

    samples = parse_ave_split(split_path)

    # 1) 过滤掉缺失音频 / 缺失视频的样本（重点：音频文件名是 clip_id.wav）
    audio_records = []
    for s in samples:
        audio_abs = os.path.join(audio_root, f"{s['clip_id']}.wav")  # ✅ FIX
        video_abs = os.path.join(video_root, f"{s['video_id']}.mp4")
        if not os.path.isfile(audio_abs):
            continue
        if not os.path.isfile(video_abs):
            continue
        audio_records.append(
            {
                "category": s["category"],
                "video_id": s["video_id"],
                "clip_id": s["clip_id"],
                "audio_path": audio_abs,
            }
        )

    # 2) 候选池：所有 unique video_id（从有效样本里取，最稳）
    cand_names = sorted(list({r["video_id"] for r in audio_records}))
    vid2idx = {vid: i for i, vid in enumerate(cand_names)}

    # 3) 候选文本（带 video token）
    cand_inst = process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)
    cand_text = [cand_inst for _ in cand_names]

    # 4) 候选视频 -> N 帧（PIL.Image），MSRVTT风格
    cand_video = []
    for vid in cand_names:
        video_path = os.path.join(video_root, f"{vid}.mp4")
        frame_dir = os.path.join(frame_root, vid)

        # 抽帧
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        frames: List[Image.Image] = []
        for p in frame_paths:
            if os.path.exists(p):
                frames.append(Image.open(p).convert("RGB"))
            else:
                frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        # 保证长度
        if len(frames) < num_frames:
            frames += [Image.new("RGB", (224, 224), (0, 0, 0))] * (num_frames - len(frames))
        cand_video.append(frames)

    # 5) 构造 query 部分
    query_texts = []
    query_audios = []
    dataset_infos = []

    for r in audio_records:
        # query text（你们统一用 build_query_text("AVE")）
        qt = build_query_text("AVE")
        assert isinstance(qt, list) and len(qt) == 1 and isinstance(qt[0], str) and qt[0].strip()
        query_texts.append(qt)

        # query audio
        query_audios.append([{"path": r["audio_path"], "bytes": None}])

        # label
        lid = vid2idx[r["video_id"]]
        dataset_infos.append(
            {
                "label_cand_id": lid,
                "label_name": r["video_id"],
                "cand_names": cand_names,
                "query_id": r["clip_id"],
                "corpus_id": r["video_id"],
                "category": r["category"],
            }
        )

    bs = len(query_texts)
    dataset = datasets.Dataset.from_dict(
        {
            "query_text": query_texts,
            "query_image": [[None]] * bs,
            "query_audio": query_audios,
            "cand_text": [cand_text] * bs,
            "cand_video": [cand_video] * bs,
            "dataset_infos": dataset_infos,
        }
    )

    corpus = None
    return dataset, corpus