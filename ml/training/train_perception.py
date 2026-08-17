#!/usr/bin/env python3
"""Train the perception CNN on synthetic camera-event frames.

Usage:
    python3 ml/training/train_perception.py --version v1 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perception"))

from synthetic_frames import generate_frames  # noqa: E402
from model import PerceptionCNN  # noqa: E402


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    X_train, y_train = generate_frames(args.train_size, seed=args.seed)
    X_val, y_val = generate_frames(args.val_size, seed=args.seed + 1000)

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)
    Xv = torch.tensor(X_val)
    yv = torch.tensor(y_val)

    model = PerceptionCNN()
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
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={epoch_loss/n:.4f}  val_acc={val_acc:.4f}")

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).argmax(1).numpy()
    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, val_pred)), 4),
        "macro_f1": round(float(f1_score(y_val, val_pred, average="macro")), 4),
        "train_seconds": round(train_time, 2),
        "train_size": args.train_size,
        "val_size": args.val_size,
    }

    out_dir = pathlib.Path(args.output_dir) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_fp32.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + metrics to {out_dir}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "models" / "perception"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
