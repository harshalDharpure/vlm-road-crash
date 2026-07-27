"""Temporal frame selection and collage for multi-frame VLM inference."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def select_key_frames(frames: List[np.ndarray], num_key: int = 4) -> List[np.ndarray]:
    if not frames:
        return []
    if len(frames) <= num_key:
        return frames
    indices = np.linspace(0, len(frames) - 1, num_key, dtype=int)
    return [frames[i] for i in indices]


def frames_to_collage(
    frames: List[np.ndarray],
    cell_size: Tuple[int, int] = (336, 336),
    max_frames: int = 4,
) -> Image.Image:
    """Build a left-to-right temporal collage (or 2x2 grid if 4 frames)."""
    key = select_key_frames(frames, max_frames)
    resized = [
        cv2.cvtColor(
            cv2.resize(f, cell_size, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        for f in key
    ]
    n = len(resized)
    if n == 1:
        return Image.fromarray(resized[0])

    if n <= 2:
        cols, rows = n, 1
    elif n <= 4:
        cols, rows = 2, 2
    else:
        cols = min(4, n)
        rows = int(np.ceil(n / cols))

    w, h = cell_size
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(resized):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return Image.fromarray(canvas)


def collage_temporal_prompt_suffix(num_key: int) -> str:
    return (
        f" The input image is a temporal collage of {num_key} sequential video frames "
        "(ordered left-to-right, top-to-bottom). Use all visible frames to describe the crash chronologically."
    )
