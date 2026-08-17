#!/usr/bin/env python3
"""Train the driving-event classifier on synthetic telemetry and save:
  - the FP32 model weights (models/event_classifier/<version>/model_fp32.pt)
  - a feature normalization config (needed identically at inference time)
  - a metrics.json (accuracy, macro-F1) consumed by evaluation/registration

Usage:
    python3 ml/training/train_event_classifier.py --version v1 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from synthetic_data import generate_driving_events  # noqa: E402
from event_classifier import EventClassifier  # noqa: E402


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    X_train, y_train = generate_driving_events(args.train_size, seed=args.seed)
    X_val, y_val = generate_driving_events(args.val_size, seed=args.seed + 1000)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    Xt = torch.tensor(X_train_n, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val_n, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    model = EventClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 64
    n = Xt.shape[0]
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv)
                val_acc = (val_logits.argmax(1) == yv).float().mean().item()
            print(f"epoch {epoch+1:3d}/{args.epochs}  loss={epoch_loss/n:.4f}  val_acc={val_acc:.4f}")

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        val_logits = model(Xv)
        val_pred = val_logits.argmax(1).numpy()
    metrics = compute_metrics(yv.numpy(), val_pred)
    metrics["train_seconds"] = round(train_time, 2)
    metrics["train_size"] = args.train_size
    metrics["val_size"] = args.val_size

    out_dir = pathlib.Path(args.output_dir) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_fp32.pt")
    with open(out_dir / "normalization.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + metrics to {out_dir}")
    print(json.dumps(metrics, indent=2))
    return metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro")), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "models" / "event_classifier"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
