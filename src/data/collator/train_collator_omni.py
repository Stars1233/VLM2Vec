from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple
import io
import random
import torch
import torchaudio
from PIL import Image
from transformers import ProcessorMixin

@dataclass
class OmniAutoProcessorCollator:
    processor: ProcessorMixin
    data_args: Any
    model_args: Any
    training_args: Any
    batch_size: Optional[int] = None

    # ---------- helpers ----------
    def _clean_image_list(self, imgs):
        """Remove None frames; return None if empty."""
        if imgs is None:
            return None
        if isinstance(imgs, list):
            imgs = [im for im in imgs if im is not None]
            return imgs if len(imgs) > 0 else None
        return imgs  # single PIL or None

    def _split_visual(self, visual):
        """
        Split visual into image or video.
        - image: PIL.Image or list length 1
        - video: list length >1
        Returns (image, video).
        """
        if visual is None:
            return None, None
        if isinstance(visual, list):
            visual = [v for v in visual if v is not None]
            if not visual:
                return None, None
            if len(visual) > 1:
                return None, visual
            return [visual[0]], None
        return [visual], None

    def _random_window(self, items, max_frames: int):
        if items is None or not isinstance(items, list):
            return items
        if max_frames <= 0 or len(items) <= max_frames:
            return items
        start = random.randint(0, len(items) - max_frames)
        return items[start : start + max_frames]

    def _load_image_from_dict(self, raw_images: dict, example: dict):
        """
        raw_images format assumed similar to your current code:
          - 'resolutions' list determines num images/frames
          - 'bytes' or 'paths' optional lists
        """
        if not isinstance(raw_images, dict) or "resolutions" not in raw_images:
            return None
        visual = []
        num_images = len(raw_images["resolutions"])
        for i in range(num_images):
            b = raw_images.get("bytes", [None]*num_images)[i] if "bytes" in raw_images else None
            p = raw_images.get("paths", [None]*num_images)[i] if "paths" in raw_images else None

            if b is not None:
                im = Image.open(io.BytesIO(b)).convert("RGB")
            elif p is not None:
                with Image.open(p) as img:
                    im = img.convert("RGB")
            else:
                im = None
            visual.append(im)

        # optional video frame subsample
        max_frames = getattr(self.data_args, "video_max_frames", 0) or 0
        if max_frames > 0:
            visual = self._random_window(visual, max_frames)

        # optional resize each frame to square
        frame_size = getattr(self.data_args, "video_frame_size", None)
        if frame_size:
            visual = [(im.resize((frame_size, frame_size)) if im is not None else None) for im in visual]

        return self._clean_image_list(visual)

    def _load_audio_batch(self, audio_items: List[Any]) -> Tuple[List[Optional[torch.Tensor]], int]:
        target_sr = int(getattr(self.data_args, "audio_sample_rate", 16000) or 16000)
        min_audio_samples = getattr(self.data_args, "audio_min_samples", None)
        if min_audio_samples is None:
            min_audio_samples = int(target_sr * 0.025)  # 25ms

        max_audio_seconds = getattr(self.data_args, "audio_max_seconds", None)
        if max_audio_seconds is not None:
            max_audio_samples = int(float(max_audio_seconds) * target_sr)
        else:
            max_audio_frames = int(getattr(self.data_args, "audio_max_frames", 1024))
            max_audio_samples = max_audio_frames * 160

        out = []
        for item in audio_items:
            if item is None:
                out.append(None)
                continue

            # HF dataset style: {"array":..., "sampling_rate":...}
            if isinstance(item, dict) and "array" in item:
                wav = torch.tensor(item["array"], dtype=torch.float32)
                sr = int(item.get("sampling_rate", target_sr))
                if wav.ndim > 1:
                    wav = wav.mean(0)
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr)
                if wav.numel() < min_audio_samples:
                    out.append(None)
                    continue
                if wav.numel() > max_audio_samples:
                    start = random.randint(0, wav.numel() - max_audio_samples)
                    wav = wav[start : start + max_audio_samples]
                out.append(wav)
                continue

            # Tensor wav
            if isinstance(item, torch.Tensor):
                wav = item.float()
                if wav.ndim > 1:
                    wav = wav.mean(0)
                if wav.numel() < min_audio_samples:
                    out.append(None)
                    continue
                if wav.numel() > max_audio_samples:
                    start = random.randint(0, wav.numel() - max_audio_samples)
                    wav = wav[start : start + max_audio_samples]
                out.append(wav)
                continue

            # Dict with path/bytes (+ optional start/end)
            if isinstance(item, dict):
                a_path = item.get("path") or item.get("audio_path") or item.get("video_path")
                a_bytes = item.get("bytes", None)
                start_t = float(item.get("start", 0.0) or 0.0)
                end_v = item.get("end", None)
                end_t = float(end_v) if end_v is not None else None

                if a_bytes is not None:
                    wave, sr = torchaudio.load(io.BytesIO(a_bytes))
                elif a_path:
                    info = torchaudio.info(a_path)
                    sr = info.sample_rate
                    frame_offset = int(start_t * sr)
                    num_frames = int((end_t - start_t) * sr) if end_t is not None else -1
                    if num_frames == 0:
                        out.append(None)
                        continue
                    wave, _ = torchaudio.load(a_path, frame_offset=frame_offset, num_frames=num_frames)
                else:
                    out.append(None)
                    continue

                if wave.numel() < min_audio_samples:
                    out.append(None)
                    continue
                if wave.ndim > 1:
                    wave = wave.mean(0)
                if sr != target_sr:
                    wave = torchaudio.functional.resample(wave, sr, target_sr)
                if wave.numel() > max_audio_samples:
                    start = random.randint(0, wave.numel() - max_audio_samples)
                    wave = wave[start : start + max_audio_samples]
                out.append(wave)
                continue

            raise ValueError(f"Unsupported audio item type: {type(item)}")

        return out, target_sr

    def _extract_raw(self, examples: List[dict], text_key: str, image_key: str, audio_key: Optional[str]):
        texts, images, audios = [], [], []
        for ex in examples:
            if ex is None or not ex:
                texts.append(" ")
                images.append(None)
                audios.append(None)
                continue

            t = ex.get(text_key, " ")
            raw_img = ex.get(image_key, None)
            raw_aud = ex.get(audio_key, None) if audio_key else None

            # if list wrappers exist
            if isinstance(t, list):
                t = t[0] if len(t) > 0 else " "
            if isinstance(raw_img, list):
                raw_img = raw_img[0] if len(raw_img) > 0 else None
            if isinstance(raw_aud, list):
                raw_aud = raw_aud[0] if len(raw_aud) > 0 else None

            # normalize image/video
            if isinstance(raw_img, dict):
                img = self._load_image_from_dict(raw_img, ex)
            elif isinstance(raw_img, list):
                img = self._clean_image_list(raw_img)
            else:
                img = raw_img  # PIL or None

            texts.append(t)
            images.append(img)
            audios.append(raw_aud)

        return texts, images, audios

    def _sig(self, img, vid, aud):
        has_a = aud is not None
        has_i = img is not None
        has_v = vid is not None
        return (has_i, has_v, has_a)

    def _process_group(self, texts, images, videos, audios, max_length: int):
        kwargs = dict(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        has_images = any(im is not None for im in images)
        has_videos = any(v is not None for v in videos)
        if has_images and has_videos:
            raise ValueError("OmniAutoProcessorCollator: cannot mix images and videos in the same group.")
        if has_images:
            kwargs["images"] = images
        if has_videos:
            kwargs["videos"] = videos
        if any(a is not None for a in audios):
            kwargs["audios"] = audios
            kwargs["sampling_rate"] = int(getattr(self.data_args, "audio_sample_rate", 16000) or 16000)

        return self.processor(**kwargs)

    # ---------- main ----------
    def __call__(self, examples: List[dict]):
        # fixed batch size check
        if self.batch_size is not None and len(examples) < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got {len(examples)}")

        # raw
        q_texts, q_imgs_raw, q_auds = self._extract_raw(examples, "query_text", "query_image", "query_audio")
        p_texts, p_imgs_raw, p_auds = self._extract_raw(examples, "pos_text", "pos_image", "pos_audio")

        # decode audio tensors only for those that exist (avoid dummy audio)
        q_wavs, q_sr = self._load_audio_batch(q_auds)
        p_wavs, p_sr = self._load_audio_batch(p_auds)

        q_imgs, q_vids = [], []
        p_imgs, p_vids = [], []
        for qi, pi in zip(q_imgs_raw, p_imgs_raw):
            q_img, q_vid = self._split_visual(qi)
            p_img, p_vid = self._split_visual(pi)
            q_imgs.append(q_img)
            q_vids.append(q_vid)
            p_imgs.append(p_img)
            p_vids.append(p_vid)

        # valid mask: any sample with audio specified but failed to load -> invalid
        valid = []
        for qa, qw, pa, pw in zip(q_auds, q_wavs, p_auds, p_wavs):
            ok = True
            if qa is not None and qw is None:
                ok = False
            if pa is not None and pw is None:
                ok = False
            valid.append(ok)

        # replace invalid samples with pure text dummy (keeps batch size stable)
        for i, ok in enumerate(valid):
            if not ok:
                q_texts[i], p_texts[i] = " ", " "
                q_imgs[i], p_imgs[i] = None, None
                q_vids[i], p_vids[i] = None, None
                q_wavs[i], p_wavs[i] = None, None

        # group by modality signature to keep processor assumptions clean
        idxs = list(range(len(examples)))
        groups = {}
        for i in idxs:
            s = (
                self._sig(q_imgs[i], q_vids[i], q_wavs[i]),
                self._sig(p_imgs[i], p_vids[i], p_wavs[i]),
            )
            groups.setdefault(s, []).append(i)

        max_len = int(getattr(self.data_args, "max_len", 256))

        def _merge(out_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # out_list already per-group; we need reconstruct per-example order.
            # We'll build dict of lists then stack/pad is already done by processor.
            merged = {}
            for out in out_list:
                for k, v in out.items():
                    merged.setdefault(k, []).append(v)
            # concatenate along batch dim
            final = {}
            for k, chunks in merged.items():
                if isinstance(chunks[0], torch.Tensor):
                    final[k] = torch.cat(chunks, dim=0)
                else:
                    # e.g. lists (rare from processor), keep concatenation
                    tmp = []
                    for c in chunks:
                        tmp.extend(c)
                    final[k] = tmp
            return final

        # process qry/pos in matching grouping to preserve alignment
        qry_outs = []
        pos_outs = []
        order_chunks = []

        for s, sub in groups.items():
            # keep sub order stable
            sub_q_text = [q_texts[i] for i in sub]
            sub_q_img  = [q_imgs[i] for i in sub]
            sub_q_vid  = [q_vids[i] for i in sub]
            sub_q_wav  = [q_wavs[i] for i in sub]

            sub_p_text = [p_texts[i] for i in sub]
            sub_p_img  = [p_imgs[i] for i in sub]
            sub_p_vid  = [p_vids[i] for i in sub]
            sub_p_wav  = [p_wavs[i] for i in sub]

            q_proc = self._process_group(sub_q_text, sub_q_img, sub_q_vid, sub_q_wav, max_len)
            p_proc = self._process_group(sub_p_text, sub_p_img, sub_p_vid, sub_p_wav, max_len)

            qry_outs.append(q_proc)
            pos_outs.append(p_proc)
            order_chunks.append(sub)

        # merge
        qry_batch = _merge(qry_outs)
        pos_batch = _merge(pos_outs)

        # attach metadata
        qry_batch["valid_example_mask"] = torch.tensor(valid, dtype=torch.bool)
        pos_batch["valid_example_mask"] = torch.tensor(valid, dtype=torch.bool)

        # for debug / hash you can still attach raw texts if you want
        qry_batch["text"] = q_texts
        pos_batch["text"] = p_texts

        return qry_batch, pos_batch
