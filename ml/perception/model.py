"""Perception: a small CNN for coarse obstacle/event detection on synthetic
camera-event frames (see ml/perception/synthetic_frames.py). TeslaEdge does
not have access to a real vehicle camera dataset, so this operates on
procedurally generated grayscale frames — documented explicitly in the
README rather than presented as real perception data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 3  # clear, obstacle_far, obstacle_near


class PerceptionCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
