"""Perception: a small CNN for image classification.

Used two ways in this repo:
  - Synthetic obstacle detection (ml/perception/synthetic_frames.py): 1
    channel, 32x32, 3 classes. The original rapid-prototyping path.
  - Real traffic-sign classification (ml/perception/real_dataset.py, the
    FTSC dataset): 3 channels (RGB), 64x64, 6 classes.

Same architecture, different input shape/channel count/class count — see
docs/architecture.md#dataset-methodology for why both exist side by side
rather than one replacing the other.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 3  # clear, obstacle_far, obstacle_near (synthetic task default)


class PerceptionCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1, image_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # image_size -> image_size/2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # image_size/2 -> image_size/4
        )
        pooled = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * pooled * pooled, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
