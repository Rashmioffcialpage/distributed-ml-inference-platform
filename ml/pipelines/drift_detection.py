#!/usr/bin/env python3
"""Drift detection for the driving-event classifier.

Compares a reference feature distribution (the training set) against a
"live" window of feature vectors — in this repo, sampled from a
deliberately shifted synthetic generator, standing in for a real change in
fleet behavior (e.g. a new region, a firmware change altering sensor
scaling) — using a per-feature two-sample Kolmogorov-Smirnov test, plus a
simple population-stability index (PSI) on the model's output distribution.

This is a batch script, not a service: in the full lifecycle it runs on a
schedule (cron / Airflow / a Kubernetes CronJob) against real production
telemetry sampled from Kafka/Redis, writes its verdict to
drift_report.json, and retrain_and_promote.py reads that verdict to decide
whether to kick off retraining.

Usage:
    python3 ml/pipelines/drift_detection.py --version v1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "common"))
sys.path.insert(0, str(ROOT / "ml" / "training"))

from synthetic_data import generate_driving_events, FEATURE_NAMES  # noqa: E402
from event_classifier import EventClassifier  # noqa: E402

KS_ALPHA = 0.01  # reject null (same distribution) if p-value < KS_ALPHA
PSI_WARN = 0.1
PSI_ALERT = 0.25


def shifted_distribution(n: int, seed: int) -> np.ndarray:
    """Simulate a distribution shift: e.g. a fleet subset now drives faster
    and brakes harder on average (as if operating on a different road type)."""
    rng = np.random.default_rng(seed)
    X, _ = generate_driving_events(n, seed=seed)
    X[:, 0] += rng.normal(15, 3, n)  # speed shifted up
    X[:, 1] -= rng.normal(1.0, 0.3, n)  # more negative (harder) acceleration
    return X


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_counts / max(len(expected), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def run(args: argparse.Namespace) -> dict:
    reference, _ = generate_driving_events(args.reference_size, seed=1)
    live = shifted_distribution(args.live_size, seed=args.seed) if args.simulate_shift else generate_driving_events(args.live_size, seed=999)[0]

    feature_report = {}
    any_drift = False
    for i, name in enumerate(FEATURE_NAMES):
        stat, pvalue = ks_2samp(reference[:, i], live[:, i])
        drifted = bool(pvalue < KS_ALPHA)
        any_drift = any_drift or drifted
        feature_report[name] = {"ks_statistic": round(float(stat), 4), "p_value": round(float(pvalue), 6), "drifted": drifted}

    # Output-distribution PSI: run the current production model over both
    # windows and compare its predicted-class distribution.
    model_dir = ROOT / "models" / "event_classifier" / args.version
    model = EventClassifier()
    model.load_state_dict(torch.load(model_dir / "model_fp32.pt", map_location="cpu"))
    model.eval()
    with open(model_dir / "normalization.json") as f:
        norm = json.load(f)
    mean, std = np.array(norm["mean"]), np.array(norm["std"])

    with torch.no_grad():
        ref_pred = model(torch.tensor((reference - mean) / std, dtype=torch.float32)).argmax(1).numpy()
        live_pred = model(torch.tensor((live - mean) / std, dtype=torch.float32)).argmax(1).numpy()
    psi = population_stability_index(ref_pred.astype(float), live_pred.astype(float), bins=4)

    verdict = "drift_detected" if (any_drift or psi > PSI_WARN) else "stable"

    report = {
        "model_version": args.version,
        "feature_drift": feature_report,
        "output_psi": round(psi, 4),
        "psi_thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "verdict": verdict,
        "recommend_retrain": verdict == "drift_detected",
    }

    out_path = ROOT / "benchmarks" / "drift_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--reference-size", type=int, default=5000)
    p.add_argument("--live-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--simulate-shift", action="store_true", default=True)
    p.add_argument("--no-simulate-shift", dest="simulate_shift", action="store_false")
    run(p.parse_args())


if __name__ == "__main__":
    main()
