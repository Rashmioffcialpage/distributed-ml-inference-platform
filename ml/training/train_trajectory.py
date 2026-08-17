#!/usr/bin/env python3
"""Train the trajectory predictor on synthetic vehicle paths.

Usage:
    python3 ml/training/train_trajectory.py --version v1 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "trajectory"))

from synthetic_data import generate_trajectories  # noqa: E402
from model import TrajectoryPredictor  # noqa: E402


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    X_train, y_train = generate_trajectories(args.train_size, seq_len=args.seq_len, seed=args.seed)
    X_val, y_val = generate_trajectories(args.val_size, seq_len=args.seq_len, seed=args.seed + 1000)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

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
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_mse = loss_fn(val_pred, yv).item()
                val_ade = (val_pred - yv).norm(dim=1).mean().item()
            print(f"epoch {epoch+1:3d}/{args.epochs}  loss={epoch_loss/n:.5f}  val_mse={val_mse:.5f}  val_ade={val_ade:.4f}m")

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv)
        final_ade = (val_pred - yv).norm(dim=1).mean().item()
        final_fde = final_ade  # single-step prediction: ADE == FDE here

    metrics = {
        "val_ade_meters": round(final_ade, 4),
        "val_fde_meters": round(final_fde, 4),
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--train-size", type=int, default=8000)
    p.add_argument("--val-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "models" / "trajectory"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
