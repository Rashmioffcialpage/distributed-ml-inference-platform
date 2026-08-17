#!/usr/bin/env python3
"""Real FP32 / FP16 / INT8 quantization benchmark for the driving-event
classifier, run on whatever hardware this script executes on (CPU-only in
this repo's dev/CI environment — see the printed `device` field in the
output and docs/benchmarks.md for the actual measured numbers).

Per the roadmap: never invent these numbers. This script measures them.

INT4 is intentionally omitted: vanilla PyTorch has no CPU INT4 kernel
without extra native dependencies (bitsandbytes, etc. — GPU-oriented), so
running it here would just print a fabricated number. It's called out as a
documented gap, not silently skipped.

Usage:
    python3 ml/quantization/benchmark.py --version v1
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))

from synthetic_data import generate_driving_events  # noqa: E402
from event_classifier import EventClassifier  # noqa: E402


def load_model(model_dir: pathlib.Path) -> EventClassifier:
    model = EventClassifier()
    model.load_state_dict(torch.load(model_dir / "model_fp32.pt", map_location="cpu"))
    model.eval()
    return model


def model_size_bytes(model: torch.nn.Module) -> int:
    buf = pathlib.Path("/tmp") / "quant_size_probe.pt"
    torch.save(model.state_dict(), buf)
    size = buf.stat().st_size
    buf.unlink()
    return size


def measure(model: torch.nn.Module, X: torch.Tensor, y: np.ndarray, warmup: int = 20, iters: int = 500) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(X[:1])

        latencies = []
        with torch.no_grad():
            for _ in range(iters):
                idx = np.random.randint(0, X.shape[0])
                start = time.perf_counter()
                model(X[idx : idx + 1])
                latencies.append((time.perf_counter() - start) * 1000)

        preds = model(X).argmax(1).numpy()

    from sklearn.metrics import accuracy_score

    lat = np.array(latencies)
    batch_start = time.perf_counter()
    with torch.no_grad():
        model(X)
    batch_elapsed = time.perf_counter() - batch_start

    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 4),
        "mean_latency_ms": round(float(lat.mean()), 4),
        "throughput_rps_batched": round(X.shape[0] / batch_elapsed, 1),
        "model_size_kb": round(model_size_bytes(model) / 1024, 2),
    }


def run(args: argparse.Namespace) -> dict:
    model_dir = pathlib.Path(args.model_dir) / args.version
    fp32 = load_model(model_dir)

    with open(model_dir / "normalization.json") as f:
        norm = json.load(f)
    mean, std = np.array(norm["mean"]), np.array(norm["std"])

    X_raw, y = generate_driving_events(args.n, seed=999)
    X = torch.tensor((X_raw - mean) / std, dtype=torch.float32)

    results = {"device": "cpu", "torch_version": torch.__version__, "n_samples": args.n}

    print("Benchmarking FP32 (baseline)...")
    results["fp32"] = measure(copy.deepcopy(fp32), X, y)

    print("Benchmarking FP16...")
    try:
        fp16 = copy.deepcopy(fp32).half()
        X_fp16 = X.half()
        results["fp16"] = measure(fp16, X_fp16, y)
    except Exception as e:  # pragma: no cover - hardware dependent
        results["fp16"] = {"error": f"fp16 CPU inference unsupported here: {e}"}

    print("Benchmarking INT8 (dynamic quantization)...")
    int8 = torch.quantization.quantize_dynamic(copy.deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8)
    results["int8"] = measure(int8, X, y)

    results["int4"] = {
        "note": "Not benchmarked: no CPU INT4 kernel available in stock PyTorch on this hardware "
        "(would require GPU + bitsandbytes/TensorRT-LLM or similar). See docs/benchmarks.md."
    }

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quantization_{args.version}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")
    print(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--model-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "models" / "event_classifier"))
    p.add_argument("--output-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
