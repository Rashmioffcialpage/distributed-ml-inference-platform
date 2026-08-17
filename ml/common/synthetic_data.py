"""Synthetic fleet-telemetry data generation.

TeslaEdge is a portfolio project, not a system trained on real vehicle
fleets, so all training data here is procedurally generated. This module is
the single source of truth for that generation so training, evaluation and
drift-detection scripts all agree on the feature schema and label semantics.

Feature vector (5-dim, matches services/fleet-simulator-go/internal/vehicle):
    [speed_kph, accel_ms2, steering_deg, jerk, noise]

Driving-event labels (4 classes), generated from simple physical thresholds
plus Gaussian noise so the classification task is nontrivial but learnable:
    0 = normal
    1 = hard_braking   (large negative acceleration)
    2 = lane_change    (large steering angle)
    3 = near_collision  (large negative acceleration AND large steering)
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["speed_kph", "accel_ms2", "steering_deg", "jerk", "noise"]
EVENT_LABELS = ["normal", "hard_braking", "lane_change", "near_collision"]


def generate_driving_events(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate (features, labels) for the driving-event classifier."""
    rng = np.random.default_rng(seed)

    speed = rng.normal(60, 20, n).clip(0, 140)
    accel = rng.normal(0, 1.5, n)
    steering = rng.normal(0, 8, n)
    jerk = rng.normal(0, 1.0, n)
    noise = rng.normal(0, 1.0, n)

    hard_brake = accel < -3.5
    lane_change = np.abs(steering) > 18
    near_collision = hard_brake & lane_change

    # Inject a few real hard-braking / lane-change events beyond the random
    # tails so classes aren't vanishingly rare.
    force_brake = rng.random(n) < 0.12
    accel = np.where(force_brake, rng.normal(-6, 1.2, n), accel)
    force_lane = rng.random(n) < 0.12
    steering = np.where(force_lane, rng.normal(0, 22, n) + rng.choice([-1, 1], n) * 15, steering)

    hard_brake = accel < -3.5
    lane_change = np.abs(steering) > 18
    near_collision = hard_brake & lane_change

    labels = np.zeros(n, dtype=np.int64)
    labels[lane_change] = 2
    labels[hard_brake] = 1
    labels[near_collision] = 3

    features = np.stack([speed, accel, steering, jerk, noise], axis=1).astype(np.float32)
    return features, labels


def generate_trajectories(n_sequences: int, seq_len: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (x, y) trajectory histories and next-step targets.

    Each sequence is a smooth, noisy 2D path (constant-velocity + a random
    curvature term), matching what a simple onboard trajectory predictor
    would consume: `seq_len` past positions -> next position.

    Returns:
        history: (n_sequences, seq_len, 2) past positions
        target:  (n_sequences, 2) next position
    """
    rng = np.random.default_rng(seed + 1)

    headings = rng.uniform(0, 2 * np.pi, n_sequences)
    speeds = rng.uniform(5, 25, n_sequences)  # m/s
    curvature = rng.normal(0, 0.05, n_sequences)

    t = np.arange(seq_len + 1)
    history = np.zeros((n_sequences, seq_len + 1, 2), dtype=np.float32)
    for i in range(n_sequences):
        theta = headings[i] + curvature[i] * t
        dx = speeds[i] * np.cos(theta) * 0.1
        dy = speeds[i] * np.sin(theta) * 0.1
        xy = np.cumsum(np.stack([dx, dy], axis=1), axis=0)
        xy += rng.normal(0, 0.05, xy.shape)
        history[i] = xy

    return history[:, :-1, :], history[:, -1, :]
