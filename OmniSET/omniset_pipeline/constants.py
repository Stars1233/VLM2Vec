"""Shared constants for OmniSET visualization pipeline."""

from __future__ import annotations

from typing import List, Tuple

MODALITY_ORDER: Tuple[str, ...] = ("t", "i", "v", "a")
MODALITY_ORDER_LONG: Tuple[str, ...] = ("text", "image", "video", "audio")

MODALITY_NAME = {
    "t": "Text",
    "i": "Image",
    "v": "Video",
    "a": "Audio",
    "text": "Text",
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
}

MODALITY_COLOR = {
    "t": "tab:blue",
    "i": "tab:orange",
    "v": "tab:green",
    "a": "tab:red",
    "text": "tab:blue",
    "image": "tab:orange",
    "video": "tab:green",
    "audio": "tab:red",
}

SOURCE_TO_SHORT = {
    "text": "t",
    "image": "i",
    "video": "v",
    "audio": "a",
}

SHORT_TO_SOURCE = {v: k for k, v in SOURCE_TO_SHORT.items()}


def build_directions(modalities: Tuple[str, ...] = MODALITY_ORDER) -> List[Tuple[str, str]]:
    """Return all cross-modal directions with source != target."""
    return [(src, tgt) for src in modalities for tgt in modalities if src != tgt]
