#!/usr/bin/env python3
"""Standalone evaluation: load a saved model version and score it against a
fresh synthetic holdout set. Used by CI and by the model registry to
gate promotion (see ml/pipelines/retrain_and_promote.py) — training and
evaluation are deliberately separate steps so a model can be re-evaluated
without retraining.

Usage:
    python3 ml/evaluation/evaluate.py --model event_classifier --version v1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "common"))


def _load_module(name: str, path: pathlib.Path):
    """Load a module by file path so same-named files (model.py in both
    ml/trajectory/ and ml/perception/) never collide in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_event_classifier(version_dir: pathlib.Path, n: int, seed: int) -> dict:
    from synthetic_data import generate_driving_events
    from sklearn.metrics import accuracy_score, f1_score

    event_classifier = _load_module("event_classifier_model", ROOT / "ml" / "training" / "event_classifier.py")
    EventClassifier = event_classifier.EventClassifier

    with open(version_dir / "normalization.json") as f:
        norm = json.load(f)
    mean, std = np.array(norm["mean"]), np.array(norm["std"])

    model = EventClassifier()
    model.load_state_dict(torch.load(version_dir / "model_fp32.pt", map_location="cpu"))
    model.eval()

    X, y = generate_driving_events(n, seed=seed)
    Xn = torch.tensor((X - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        pred = model(Xn).argmax(1).numpy()

    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "n_samples": n,
    }


def evaluate_trajectory(version_dir: pathlib.Path, n: int, seed: int) -> dict:
    from synthetic_data import generate_trajectories

    trajectory_model = _load_module("trajectory_model", ROOT / "ml" / "trajectory" / "model.py")
    model = trajectory_model.TrajectoryPredictor()
    model.load_state_dict(torch.load(version_dir / "model_fp32.pt", map_location="cpu"))
    model.eval()

    X, y = generate_trajectories(n, seed=seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        pred = model(Xt)
        ade = (pred - yt).norm(dim=1).mean().item()

    return {"val_ade_meters": round(ade, 4), "n_samples": n}


def evaluate_perception(version_dir: pathlib.Path, n: int, seed: int) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    synthetic_frames = _load_module("synthetic_frames", ROOT / "ml" / "perception" / "synthetic_frames.py")
    generate_frames = synthetic_frames.generate_frames
    perception_model = _load_module("perception_model", ROOT / "ml" / "perception" / "model.py")
    model = perception_model.PerceptionCNN()
    model.load_state_dict(torch.load(version_dir / "model_fp32.pt", map_location="cpu"))
    model.eval()

    X, y = generate_frames(n, seed=seed)
    with torch.no_grad():
        pred = model(torch.tensor(X)).argmax(1).numpy()

    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "n_samples": n,
    }


EVALUATORS = {
    "event_classifier": evaluate_event_classifier,
    "trajectory": evaluate_trajectory,
    "perception": evaluate_perception,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=EVALUATORS.keys())
    p.add_argument("--version", default="v1")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    version_dir = ROOT / "models" / args.model / args.version
    metrics = EVALUATORS[args.model](version_dir, args.n, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
