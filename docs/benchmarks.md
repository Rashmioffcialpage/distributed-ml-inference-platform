# TeslaEdge Benchmarks

Every number on this page was produced by running the scripts in this repo
(`ml/training`, `ml/quantization/benchmark.py`, `ml/pipelines/drift_detection.py`,
`benchmarks/loadtest`) — none of it is invented. Reproduce any table below with
the commands shown in its section header. See [Environment](#environment) for
the exact hardware these numbers came from; re-running on different hardware
(especially a GPU) will produce different absolute numbers, though the
relative shape (INT8 smaller/similar-accuracy, more workers = more
throughput) should hold.

## Environment

All numbers below were collected on a CPU-only Linux container (no GPU
available in this environment): 4 vCPUs, PyTorch 2.13.0 (CPU build). A real
GPU deployment would show materially different — and generally much better
— absolute latency/throughput numbers, particularly for FP16, which has no
efficient CPU kernel path. This gap is documented rather than papered over.

## Model training results

Reproduce: `python3 ml/training/train_<model>.py --version v1`

| Model | Task | Metric | Value | Train time | Train / val size |
|---|---|---|---|---|---|
| `event_classifier` | 4-class driving-event classification | accuracy / macro-F1 | 0.996 / 0.982 | 8.1s | 20,000 / 4,000 |
| `trajectory` | next-position regression (LSTM) | val ADE (avg displacement error) | 0.42 m | 4.3s | 8,000 / 1,500 |
| `perception` | 3-class obstacle detection (CNN) | accuracy / macro-F1 | 1.000 / 1.000 | 5.7s | 6,000 / 1,200 |

All three train on **procedurally generated synthetic data** (see
`ml/common/synthetic_data.py` and `ml/perception/synthetic_frames.py`) —
TeslaEdge has no real vehicle fleet or camera dataset. The perception task's
perfect score reflects that the synthetic frames are, by design, an easy
signal-detection problem (a Gaussian blob vs. background noise); it
demonstrates the training/eval/registry pipeline working end to end, not a
claim about real-world perception accuracy. See
[Dataset methodology](architecture.md#dataset-methodology) for why synthetic
data was the right call for a portfolio project and what a real deployment
would need instead.

## Quantization benchmark (FP32 / FP16 / INT8)

Reproduce: `python3 ml/quantization/benchmark.py --version v1`
(raw output saved to `benchmarks/quantization_v1.json`)

Model: `event_classifier` v1 (a 3-layer, ~1.4K-parameter MLP), 2,000 synthetic
holdout samples, single-sample (unbatched) latency unless noted.

| Precision | Accuracy | p50 latency | p95 latency | Model size | Batched throughput |
|---|---|---|---|---|---|
| FP32 (baseline) | 0.9935 | 0.038 ms | 0.057 ms | 8.50 KB | 2,173,108 req/s |
| FP16 | 0.9935 | 0.042 ms | 0.073 ms | 5.81 KB | 2,170,770 req/s |
| INT8 (dynamic) | 0.9945 | 0.108 ms | 0.163 ms | 6.59 KB | 3,815,694 req/s |
| INT4 | not benchmarked | — | — | — | — |

**Findings, honestly reported:**

- **INT8 was *slower* per single-sample call than FP32 on this CPU**, despite
  being smaller. For a model this tiny (three `nn.Linear` layers), the
  quantize/dequantize overhead per call dominates — there simply isn't enough
  matmul work per request for INT8's cheaper arithmetic to pay for itself.
  INT8's batched throughput is nonetheless ~1.76x FP32's, because that
  overhead amortizes across a batch. **The lesson: quantization benefits
  scale with model size and batch size — a 1.4K-parameter classifier is not
  where you'd expect INT8 to win on latency, and the numbers here confirm
  that rather than assuming it.**
- FP16 showed no meaningful latency or accuracy change on CPU (expected: x86
  CPUs largely emulate FP16 arithmetic in FP32 rather than running it
  natively; the benefit is real on GPU tensor cores, not measurable here).
- **INT4 is not benchmarked** — stock PyTorch has no CPU INT4 kernel; it
  requires GPU-oriented libraries (bitsandbytes, TensorRT-LLM) not available
  in this environment. Reporting a number here without a real kernel to back
  it would be exactly the kind of invented figure the roadmap warns against.

## Load test results

Reproduce: `go run ./benchmarks/loadtest -url http://localhost:8080 -concurrency 50 -duration 20s -api-key devkey`

End-to-end load against the full path (`loadtest` → gateway → router (gRPC) →
Python/PyTorch inference worker over HTTP), FP32 `event_classifier`, 50
concurrent clients, 20 second run:

| Workers | Total requests | Success rate | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 1 | 9,614 | 100% | 480.7 req/s | 98.96 ms | 149.98 ms | 207.97 ms |
| 2 | 12,119 | 100% | 606.0 req/s | 89.37 ms | 134.17 ms | 182.84 ms |

Adding a second inference worker raised throughput **+26%** and lowered p50
latency **~10%** — real evidence that the router's load-aware worker
selection (`services/router-go/internal/routing/selector.go`) actually
distributes work rather than pinning it to one worker. The end-to-end
latency here (~90-100ms) is dominated by Python/uvicorn per-request overhead
and this environment's CPU contention, not the ~0.04-0.1ms model inference
time measured in isolation above — that gap is itself a useful finding: the
serving stack, not the model, is the bottleneck at this scale, which is
exactly the kind of thing an ML infra engineer should be able to point at
and explain.

Zero failures across ~21,700 requests in both runs.

## Retry / fault tolerance

`services/scheduler-go/internal/worker/dispatcher_test.go` exercises this
directly rather than only asserting it in prose: a router that always fails
causes the dispatcher to retry with linear backoff up to `MaxAttempts`, then
publish the job to `<topic>.dlq` — verified as 3 router calls followed by
exactly one dead-lettered message.

## Drift detection

Reproduce: `python3 ml/pipelines/drift_detection.py --version v1`
(output saved to `benchmarks/drift_report.json`)

Reference window: 5,000 training-distribution samples. "Live" window: 2,000
samples from a simulated distribution shift (speed +15 km/h, harder average
braking — standing in for e.g. a fleet subset now driving a different road
type). Two-sample Kolmogorov-Smirnov test per feature, α = 0.01:

| Feature | KS statistic | p-value | Drifted? |
|---|---|---|---|
| speed_kph | 0.29 | < 0.001 | **yes** |
| accel_ms2 | 0.26 | < 0.001 | **yes** |
| steering_deg | 0.03 | 0.135 | no |
| jerk | 0.02 | 0.442 | no |
| noise | 0.02 | 0.471 | no |

Verdict: `drift_detected`, `recommend_retrain: true` — correctly flagging
the two features that were actually shifted and correctly ignoring the three
that weren't.

## Retrain-and-promote pipeline

Reproduce: `python3 ml/pipelines/retrain_and_promote.py --new-version v2 --deployer-url http://localhost:8081`
(requires `model-deployer` + Postgres running)

Verified end to end against a live `model-deployer-go` + Postgres instance:
trains a candidate, evaluates it on a fresh holdout set, registers it via the
registry API, compares its accuracy against the current production version,
and only calls `/deploy` if it clears `--min-improvement` — otherwise it
exits non-zero and leaves the candidate in `staging` for manual review. First
run (no production version yet) promoted unconditionally, as designed; the
git history's `models/event_classifier/v2/` artifact is that run's actual
output.
