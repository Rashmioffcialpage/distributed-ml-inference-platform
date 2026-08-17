#!/usr/bin/env python3
"""TeslaEdge inference worker: loads a trained model version, serves
POST /predict (the HTTP contract router-go forwards requests to), and
self-registers/heartbeats with router-go over gRPC so the router's worker
registry always reflects which workers are actually alive.

Environment variables:
    MODEL_NAME        default "driving-event-classifier" (registry name is
                       "event_classifier"; MODEL_NAME is the routing key
                       clients request)
    MODEL_VERSION      default "v1"
    MODEL_PRECISION    fp32 | fp16 | int8   (default fp32)
    MODELS_DIR         default ../../models relative to this file
    WORKER_ID          default "worker-<random>"
    WORKER_PORT        default 8100
    WORKER_ENDPOINT    default "http://localhost:<WORKER_PORT>" (what the
                       router should call back on — override to the
                       container's DNS name in docker-compose/k8s)
    ROUTER_GRPC_ADDR   default "localhost:9090"
    GPU_AVAILABLE      default "false" (this reference worker always runs on
                       CPU; the flag exists so the router's GPU-aware
                       routing logic has something real to select on when a
                       GPU-backed worker is added)
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import threading
import time

import grpc
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "common"))
sys.path.insert(0, str(ROOT / "ml" / "training"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from event_classifier import EventClassifier  # noqa: E402
from inference.v1 import inference_pb2, inference_pb2_grpc  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "driving-event-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MODEL_PRECISION = os.environ.get("MODEL_PRECISION", "fp32")
MODELS_DIR = pathlib.Path(os.environ.get("MODELS_DIR", ROOT / "models"))
WORKER_ID = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8100"))
WORKER_ENDPOINT = os.environ.get("WORKER_ENDPOINT", f"http://localhost:{WORKER_PORT}")
ROUTER_GRPC_ADDR = os.environ.get("ROUTER_GRPC_ADDR", "localhost:9090")
GPU_AVAILABLE = os.environ.get("GPU_AVAILABLE", "false").lower() == "true"

PRECISION_MAP = {
    "fp32": inference_pb2.PRECISION_FP32,
    "fp16": inference_pb2.PRECISION_FP16,
    "int8": inference_pb2.PRECISION_INT8,
    "int4": inference_pb2.PRECISION_INT4,
}


def load_model():
    version_dir = MODELS_DIR / "event_classifier" / MODEL_VERSION
    model = EventClassifier()
    model.load_state_dict(torch.load(version_dir / "model_fp32.pt", map_location="cpu"))
    model.eval()

    if MODEL_PRECISION == "fp16":
        model = model.half()
    elif MODEL_PRECISION == "int8":
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    with open(version_dir / "normalization.json") as f:
        norm = json.load(f)
    return model, np.array(norm["mean"]), np.array(norm["std"])


MODEL, MEAN, STD = load_model()
EVENT_LABELS = ["normal", "hard_braking", "lane_change", "near_collision"]

app = FastAPI(title="TeslaEdge Inference Worker")


class PredictRequest(BaseModel):
    features: list[float]
    precision: str = "fp32"
    model_name: str = MODEL_NAME


class PredictResponse(BaseModel):
    output: list[float]
    predicted_label: str
    confidence: float
    model_version: str
    precision_used: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    x = (np.array(req.features, dtype=np.float32) - MEAN) / STD
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    if MODEL_PRECISION == "fp16":
        tensor = tensor.half()

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        pred_idx = int(probs.argmax())

    return PredictResponse(
        output=[round(float(p), 6) for p in probs.tolist()],
        predicted_label=EVENT_LABELS[pred_idx],
        confidence=round(float(probs[pred_idx]), 6),
        model_version=MODEL_VERSION,
        precision_used=MODEL_PRECISION,
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION, "precision": MODEL_PRECISION}


def register_with_router_loop():
    """Register (and re-register as a heartbeat) with router-go every 10s."""
    while True:
        try:
            with grpc.insecure_channel(ROUTER_GRPC_ADDR) as channel:
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                stub.RegisterWorker(
                    inference_pb2.RegisterWorkerRequest(
                        worker_id=WORKER_ID,
                        supported_models=[MODEL_NAME],
                        gpu_available=GPU_AVAILABLE,
                        max_precision=PRECISION_MAP.get(MODEL_PRECISION, inference_pb2.PRECISION_FP32),
                        endpoint=WORKER_ENDPOINT,
                    ),
                    timeout=5,
                )
        except grpc.RpcError as e:
            print(f"[worker] router registration failed (will retry): {e.details() if hasattr(e, 'details') else e}")
        time.sleep(10)


@app.on_event("startup")
def start_registration_thread():
    thread = threading.Thread(target=register_with_router_loop, daemon=True)
    thread.start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
