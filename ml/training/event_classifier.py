"""Driving-event classifier: a small MLP over the 5-dim telemetry feature
vector emitted by fleet-simulator-go (speed, accel, steering, jerk, noise).

This is deliberately small (a few thousand parameters) — the point of
TeslaEdge is demonstrating the surrounding ML infrastructure (routing,
quantization, versioning, drift detection), not chasing state-of-the-art
accuracy on a synthetic dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_FEATURES = 5
NUM_CLASSES = 4


class EventClassifier(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
