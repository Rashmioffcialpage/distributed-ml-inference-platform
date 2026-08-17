#!/usr/bin/env python3
"""Retraining/promotion pipeline: the ML side of TeslaEdge's model
lifecycle (training -> evaluation -> registry -> deployment -> monitoring
-> retraining).

Given a drift report (from drift_detection.py) recommending retraining,
this script:
  1. Trains a new candidate version of the event classifier.
  2. Evaluates it against a fresh holdout set.
  3. Registers the candidate with model-deployer-go's registry API.
  4. Promotes it to production ONLY if it beats the current production
     version's accuracy by --min-improvement; otherwise it stays in
     "staging" and the script exits nonzero so a CI/CD job can flag it
     for manual review — this is the "automatic rollback if a new model
     fails defined quality thresholds" requirement from the roadmap,
     applied at promotion time rather than after a bad deploy.

Usage:
    python3 ml/pipelines/retrain_and_promote.py --model-name event_classifier \
        --new-version v2 --deployer-url http://localhost:8081
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]


def train_candidate(version: str, epochs: int) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "ml" / "training" / "train_event_classifier.py"), "--version", version, "--epochs", str(epochs)],
        check=True,
    )
    with open(ROOT / "models" / "event_classifier" / version / "metrics.json") as f:
        return json.load(f)


def evaluate_candidate(version: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(ROOT / "ml" / "evaluation" / "evaluate.py"), "--model", "event_classifier", "--version", version],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def current_production_accuracy(deployer_url: str, model_name: str) -> float | None:
    resp = requests.get(f"{deployer_url}/v1/models/{model_name}/production", timeout=5)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("metrics", {}).get("accuracy")


def register(deployer_url: str, model_name: str, version: str, artifact_path: str, metrics: dict):
    resp = requests.post(
        f"{deployer_url}/v1/models/register",
        json={"model_name": model_name, "version": version, "precision": "fp32", "artifact_path": artifact_path, "metrics": metrics},
        timeout=5,
    )
    resp.raise_for_status()


def promote(deployer_url: str, model_name: str, version: str, reason: str):
    resp = requests.post(f"{deployer_url}/v1/models/{model_name}/deploy", json={"version": version, "reason": reason}, timeout=5)
    resp.raise_for_status()


def run(args: argparse.Namespace) -> int:
    print(f"[1/4] Training candidate {args.new_version}...")
    train_candidate(args.new_version, args.epochs)

    print(f"[2/4] Evaluating {args.new_version} on fresh holdout data...")
    eval_metrics = evaluate_candidate(args.new_version)
    print(json.dumps(eval_metrics, indent=2))

    print("[3/4] Registering candidate with model-deployer...")
    artifact_path = f"models/{args.model_name}/{args.new_version}/model_fp32.pt"
    register(args.deployer_url, args.model_name, args.new_version, artifact_path, eval_metrics)

    print("[4/4] Comparing against current production version...")
    prod_accuracy = current_production_accuracy(args.deployer_url, args.model_name)
    candidate_accuracy = eval_metrics["accuracy"]

    if prod_accuracy is None:
        print("No production version exists yet; promoting candidate unconditionally.")
        promote(args.deployer_url, args.model_name, args.new_version, "initial production deployment")
        return 0

    improvement = candidate_accuracy - prod_accuracy
    print(f"production accuracy={prod_accuracy:.4f}  candidate accuracy={candidate_accuracy:.4f}  delta={improvement:+.4f}")

    if improvement >= args.min_improvement:
        promote(args.deployer_url, args.model_name, args.new_version,
                f"accuracy improved by {improvement:+.4f} (>= threshold {args.min_improvement})")
        print(f"Promoted {args.new_version} to production.")
        return 0

    print(f"Candidate did NOT clear the promotion threshold ({args.min_improvement}); "
          f"leaving it in staging for manual review.")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="event_classifier")
    p.add_argument("--new-version", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--deployer-url", default="http://localhost:8081")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
