"""Trajectory prediction: a small LSTM that consumes a history of (x, y)
positions and predicts the next position, per the roadmap's "predict future
movement from simulated historical trajectories" requirement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim: int = 2, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final timestep's hidden state
        return self.head(last)
