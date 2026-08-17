"""Synthetic 32x32 grayscale 'camera event' frames.

Class 0 (clear): background noise only.
Class 1 (obstacle_far): a small, low-contrast blob.
Class 2 (obstacle_near): a large, high-contrast blob.

This is a stand-in for real perception imagery so the perception model has
something nontrivial (but honestly synthetic) to learn from.
"""
from __future__ import annotations

import numpy as np

FRAME_SIZE = 32
CLASS_NAMES = ["clear", "obstacle_far", "obstacle_near"]


def _blob(size: int, cx: float, cy: float, radius: float, intensity: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return intensity * np.exp(-dist2 / (2 * radius**2))


def generate_frames(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 2)
    frames = np.zeros((n, 1, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    labels = rng.integers(0, 3, n)

    for i in range(n):
        frame = rng.normal(0.1, 0.05, (FRAME_SIZE, FRAME_SIZE)).clip(0, 1)
        label = labels[i]
        if label == 1:
            cx, cy = rng.uniform(8, 24, 2)
            frame += _blob(FRAME_SIZE, cx, cy, radius=3, intensity=0.35)
        elif label == 2:
            cx, cy = rng.uniform(10, 22, 2)
            frame += _blob(FRAME_SIZE, cx, cy, radius=7, intensity=0.9)
        frames[i, 0] = np.clip(frame, 0, 1)

    return frames, labels.astype(np.int64)
