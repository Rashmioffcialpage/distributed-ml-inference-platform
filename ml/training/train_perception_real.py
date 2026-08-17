#!/usr/bin/env python3
"""Train the perception CNN on FTSC — a real, public, CC BY-NC 4.0 licensed
dataset of French traffic sign photographs captured from vehicle-mounted
cameras (https://github.com/andrewcaunes/FTSC). This is the "real-world
driving dataset" evaluation referenced in the README: synthetic data
(train_perception.py) was used for rapid system validation; this script is
the real-data evaluation that followed it.

Downloads the dataset on first run (~233MB, one-time git clone) into
--data-dir, which is NOT committed to this repo (see .gitignore) — only the
code and the resulting small trained model/metrics are.

Usage:
    python3 ml/training/train_perception_real.py --version v1 --epochs 10 --limit 4000
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perception"))

from real_dataset import load_ftsc, CLASS_NAMES, IMAGE_SIZE, LICENSE_NOTE  # noqa: E402
from model import PerceptionCNN  # noqa: E402


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    print(LICENSE_NOTE)

    data_dir = pathlib.Path(args.data_dir)
    print(f"Loading up to {args.limit} FTSC images from {data_dir} (downloads on first run)...")
    X, y = load_ftsc(data_dir, limit=args.limit, image_size=args.image_size, seed=args.seed)
    print(f"Loaded {len(X)} images across {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

    n_val = max(1, int(len(X) * args.val_fraction))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X[train_idx])
    yt = torch.tensor(y[train_idx])
    Xv = torch.tensor(X[val_idx])
    yv = torch.tensor(y[val_idx])

    model = PerceptionCNN(num_classes=len(CLASS_NAMES), in_channels=3, image_size=args.image_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32
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
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    metrics = {
        "dataset": "FTSC (French Traffic Sign Classification)",
        "dataset_url": "https://github.com/andrewcaunes/FTSC",
        "dataset_license": "CC BY-NC 4.0",
        "classes": CLASS_NAMES,
        "accuracy": round(float(accuracy_score(yv.numpy(), val_pred)), 4),
        "macro_f1": round(float(f1_score(yv.numpy(), val_pred, average="macro")), 4),
        "train_seconds": round(train_time, 2),
        "train_size": int(n),
        "val_size": int(n_val),
    }
    print("\nPer-class report:")
    print(classification_report(yv.numpy(), val_pred, target_names=CLASS_NAMES, zero_division=0))

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
    p.add_argument("--limit", type=int, default=4000, help="cap on images loaded (full set is 10,959)")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "data" / "ftsc"))
    p.add_argument("--output-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "models" / "perception_real"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
