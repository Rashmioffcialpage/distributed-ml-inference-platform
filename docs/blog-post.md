# Building TeslaEdge: a distributed ML inference platform, and what actually broke

TeslaEdge is a portfolio project: a distributed inference and fleet-learning
platform modeled on the kind of system a fleet of vehicles would need to
serve ML predictions at scale — driving-event classification, trajectory
prediction, obstacle perception — with requests routed across heterogeneous
workers, model versions safely deployed and rolled back, drift detected, and
the whole thing observable end to end. It's majority Go (routing,
scheduling, gateway, model registry) with PyTorch doing exactly the part
only PyTorch should: training and running the models.

This isn't a writeup of a finished, polished system. It's a writeup of the
things that actually went wrong while building it, because those are more
useful to read than a features list. Code and full numbers:
[github.com/rashmioffcialpage/distributed-ml-inference-platform](https://github.com/rashmioffcialpage/distributed-ml-inference-platform).

## Why Go for the infrastructure, Python only for the models

The instinct with an "ML platform" is to reach for Python everywhere. That's
backwards for the parts of this system that aren't actually doing math. The
API gateway, the gRPC router that picks which worker handles a request, the
async job scheduler with retry/backoff/dead-lettering, the fleet simulator
running hundreds of concurrent vehicles, the model registry with
deploy/rollback/canary traffic splits — none of that is ML. It's concurrent
network I/O and state management, which is Go's actual strength, and it
means the inference workers (the one place PyTorch has to be) can crash,
restart, or scale independently without taking the routing layer down with
them. Router-to-worker talks plain HTTP/JSON instead of gRPC specifically so
a new worker — different precision, different hardware, someone's weekend
GPU box — only has to implement `POST /predict` and `GET /healthz` to join
the pool, no proto codegen required.

## The quantization result that went the "wrong" way

The obvious thing to expect from INT8 quantization is "smaller and faster."
Benchmarking it properly on the driving-event classifier (a ~1.4K-parameter,
3-layer MLP) showed smaller — yes, 6.59 KB vs. 8.50 KB — but for
single-sample latency, INT8 was *slower* than FP32: 0.108ms vs. 0.038ms p50.

The reason is that quantize/dequantize has fixed overhead per call, and a
model this tiny doesn't do enough matmul work per request for INT8's
cheaper arithmetic to pay that cost back. Batch it, though, and the
overhead amortizes: INT8's *batched* throughput came out to 3.82M samples/s
against FP32's 2.17M — a genuine 76% win. The honest version of this
finding is "quantization benefits scale with model size and batch size,"
and a portfolio-scale MLP is a legitimate demonstration of *why* that's
true, not just an assertion of it. Reporting the latency regression instead
of quietly only showing the throughput win was the right call — a curated
number that only shows the flattering side isn't a benchmark, it's
marketing.

FP16 told a similarly unglamorous story: no measurable benefit on CPU at
all, because x86 largely emulates FP16 in FP32 rather than running it
natively. That's expected and not a bug — it would look completely
different on a GPU with tensor cores, which this environment didn't have.
INT4 isn't benchmarked at all, because stock PyTorch has no CPU INT4 kernel
without GPU-oriented libraries (bitsandbytes, TensorRT-LLM) that weren't
available. A fabricated INT4 number would have been worse than an honest
gap in the results table.

## The dataset detour: GTSRB was the plan, until it wasn't reachable

The perception model was originally meant to train on GTSRB, a well-known
traffic-sign dataset hosted on Kaggle. The build environment's network
policy returned `403 Forbidden` on Kaggle and every Hugging Face host —
confirmed by directly testing each one, not assumed from a blocked-looking
error. GitHub-hosted data was reachable, so the perception model trains
instead on [FTSC](https://github.com/andrewcaunes/FTSC), 10,959 real
vehicle-camera photographs of French traffic signs (CC BY-NC 4.0), a
dataset that exists specifically because someone else hit the same Kaggle
problem and open-sourced their own scrape as a workaround.

That swap mattered for what it did to the numbers. The synthetic perception
task — a Gaussian blob against background noise — scored a perfect
1.000/1.000 accuracy/macro-F1, which demonstrates the training/eval/
registry pipeline works, but says nothing about real-world perception.
FTSC came back at 0.869/0.844: a real, imperfect number, with a per-class
breakdown that actually shows the ugly part (0.70 precision on a rare
"not-in-catalogue" sign class vs. 0.94 on the common regulatory-sign
class) instead of hiding it. A synthetic-only perception result would have
been the more impressive-looking number and the less honest one. A
ready-to-run GTSRB download script ships in the repo anyway
(`ml/perception/download_gtsrb_kaggle.py`) for anyone with Kaggle access
who wants to swap the real thing back in.

## Getting `docker compose up` to actually work

This is the part worth writing up in the most detail, because it's where
"the plumbing is configured" quietly diverged from "the system actually
runs," in three separate ways.

**Docker Hub itself was blocked.** Every base image pull returned `403`.
The fix wasn't retrying — it was routing through `mirror.gcr.io`, Google's
public, unauthenticated Docker Hub mirror (the same one GKE nodes use by
default). That's not a workaround specific to one sandbox; it's a real
resilience improvement against Docker Hub rate limits or outages on any
machine, so it's what the Dockerfiles use permanently now, not just for
this one build.

**The PyTorch wheel silently assumed a GPU.** Every worker in this system
runs CPU-only — there's no GPU in the docker-compose topology at all — but
the default PyPI `torch` wheel for Linux is CUDA-linked, and it eagerly
`dlopen()`s its NVIDIA shared libraries (`libcudart`, `libcublasLt`, ...) at
*import* time, not just when a GPU op actually runs. Trying to skip those
with `pip install --no-deps` to shrink the image broke `import torch`
outright with "libcublasLt.so not found" — proof that those libraries are a
real import-time requirement of that wheel, not just a GPU-only cost that
could be trimmed away. (`download.pytorch.org`'s CPU-only wheel index
avoids this entirely if it's reachable — worth using on a machine that can
reach it.) The image also carried scipy, scikit-learn, pytest, httpx, and
`grpcio-tools` — none of which the serving code imports — because they'd
been pulled in from the same requirements file used for training. Splitting
out a `requirements-serving.txt` and making the worker's Dockerfile
multi-stage (protobuf codegen in one stage, only the serving deps in the
final one) fixed both problems at once.

**The distributed tracing claim wasn't true.** OpenTelemetry was
initialized in every Go service — tracer provider set up, OTLP exporter
configured, propagators registered — and an earlier draft of this project's
README said traces "propagate Gateway → Router → Scheduler." That sentence
described the plumbing correctly and the actual behavior not at all:
nothing in the request path ever called `tracer.Start()`, so there were no
spans to export, ever, regardless of how correctly the exporter was
configured. `curl http://localhost:16686/api/services` against a live
Jaeger instance came back with an empty list — the kind of check that's
easy to skip when the code *looks* instrumented. The fix was adding actual
span creation in the gateway's `Infer` handler and the scheduler's
dispatch loop, plus `otelgrpc` client/server interceptors so a span
context genuinely propagates across the gRPC call from Gateway into
Router. After that fix, the same Jaeger query returned real traces with
real latencies — a `gateway.Infer` parent span containing the `Predict`
gRPC child span, at durations like 1.85ms and 2.32ms, not zero and not
invented.

None of these three were exotic bugs. They were the specific kind of thing
that "looks done" in a code review and isn't done at all until someone
actually runs the system and checks.

## What the numbers actually show

| | |
|---|---|
| Event classifier accuracy / macro-F1 | 0.996 / 0.982 |
| Perception on real data (FTSC) accuracy / macro-F1 | 0.869 / 0.844 |
| INT8 vs FP32 batched throughput | 3.82M vs 2.17M samples/s (+76%) |
| INT8 vs FP32 single-sample p50 latency | 0.108ms vs 0.038ms (INT8 slower, by design at this scale) |
| End-to-end load test, 2 workers | 606.0 req/s, p50 89ms, 0 failures / 12,119 requests |
| Full `docker compose up` stack | All 12 containers healthy; sync + async inference, model registry, distributed tracing verified live |

Full methodology, reproduction commands, and every caveat these numbers
carry: [docs/benchmarks.md](benchmarks.md).

## What's still honestly incomplete

The Router → Worker hop (Go to Python, over HTTP) isn't traced — a trace
currently ends at the Router because the Python worker has no OTel SDK
wired in yet. The serving worker does one synchronous forward pass per
request with no dynamic batching or warm pool. The model-promotion gate
compares offline accuracy deltas rather than running a real shadow-traffic
or canary comparison. All three are listed as what's next, not smoothed
over as already done — the same standard applied to everything else in
this writeup.
