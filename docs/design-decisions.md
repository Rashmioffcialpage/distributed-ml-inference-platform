# Design Decisions and Tradeoffs

This document records the choices that aren't obvious from reading the code,
and what was given up to make them. See [architecture.md](architecture.md)
for the system-shape rationale (sync vs. async paths, gRPC vs. HTTP, etc.);
this page is the lower-level "why this and not that" log.

## Go workspace: one shared module + five service modules

**Decision:** `go.work` ties together a `shared` module (proto, domain
models, config/logging/metrics/tracing/queue helpers) and five independently
versioned service modules under `services/`, each with its own `go.mod` and
a `replace .../shared => ../../shared` directive.

**Alternative considered:** A single monorepo Go module for everything.

**Why not:** Each service is meant to be independently buildable and
deployable (its own Dockerfile, its own container image, its own
`go build ./cmd/...`). A single module would still allow that, but a
multi-module workspace is closer to how a real multi-team platform is
organized — each service module can add its own dependencies without
bumping every other service's `go.sum`, and CI can build/test/version them
independently (see `.github/workflows/go-ci.yml`'s matrix). The cost is
some ceremony (the `replace` directive in every service's `go.mod`,
`go.work` for local dev) that a single-module repo wouldn't have.

## Router-to-worker transport: plain HTTP/JSON, not gRPC

**Decision:** The Gateway/Scheduler talk to the Router over gRPC (see
`shared/proto/inference/v1`), but the Router talks to inference workers over
plain HTTP POST with a JSON body.

**Alternative considered:** gRPC end to end, including to Python workers.

**Why not:** It would work — grpcio has fine Python support — but it means
every new worker implementation (a different framework, a different
language, someone's weekend GPU box) needs proto codegen wired up before it
can join the pool. A worker only needs to implement `POST /predict` and
`GET /healthz`; the contract is inspectable with `curl`. Given TeslaEdge's
worker pool is explicitly meant to be heterogeneous (different precisions,
potentially different frameworks), that low a bar to entry mattered more
than gRPC's efficiency on a hop that's already dominated by model inference
time, not serialization.

## Queue backend as an interface: Redis Streams default, Kafka alternate

**Decision:** `shared/pkg/queue.Queue` is a small interface
(Enqueue/Consume/DeadLetter); Redis Streams is the default backend, Kafka
(`segmentio/kafka-go`) is a drop-in alternative behind `QUEUE_BACKEND=kafka`.

**Alternative considered:** Pick one and hard-wire it.

**Why not:** The roadmap explicitly calls for "Kafka / Redis" as the
messaging layer, and in practice this is a real decision teams face —
Redis Streams is far lighter to operate for a single-region deployment;
Kafka's partitioned, replayable log earns its operational cost at
multi-consumer-group, multi-region scale. Implementing both against one
interface demonstrates understanding of *when* each is the right call,
rather than picking one and asserting it's correct. The cost: the interface
is deliberately minimal (no exactly-once semantics, no transactional
enqueue+ack) — it's shaped by what both Redis Streams and Kafka can do, not
by either one's full feature set.

## Model registry: Postgres-backed service, not embedded in the Router

**Decision:** `model-deployer-go` owns model version/stage/traffic-split
state in Postgres; the Router only holds a live, in-memory worker registry
(no persistence).

**Alternative considered:** Store everything in one place — either have the
Router own model metadata too, or have workers read Postgres directly.

**Why not:** These are different consistency and latency requirements
wearing the same word ("registry"). Deploy/rollback need transactional,
durable, auditable writes (`registry.Deploy`'s single transaction that
atomically demotes the old production version and promotes the new one, and
the `deployment_events` audit log). Inference routing needs sub-millisecond
in-memory lookups on every request and must never wait on a lock a deploy
transaction is holding. Splitting them means a slow or failed deploy can
never add latency or an error path to live inference traffic. The cost is
one more service and one more datastore to run — justified here because the
alternative (workers polling Postgres before every inference call) would put
a network+DB round trip on the hot path.

## Model serving: dynamic quantization scripts, not a model-serving framework

**Decision:** Quantization (`ml/quantization/benchmark.py`) uses
`torch.quantization.quantize_dynamic` directly; serving
(`ml/serving/server.py`) is a small FastAPI app loading a `state_dict`
checkpoint, not TorchServe/Triton/BentoML.

**Alternative considered:** Adopt a production model-serving framework.

**Why not:** Those frameworks solve problems TeslaEdge's Go layer already
solves on purpose — request routing, batching policy, multi-model
management — and adding one would duplicate that logic in two places (the
Go Router and the serving framework) or hide it inside a framework's config
instead of in reviewable code. For a project meant to *demonstrate*
understanding of routing/versioning/precision-selection, keeping that logic
visible in Go beats delegating it to a framework. The tradeoff: this
worker is not production-hardened (no dynamic batching, no model warm pool)
— explicitly named as a future improvement in the [README](../README.md#future-improvements)
rather than silently left out.

## Retry policy: linear backoff + fixed max attempts, not exponential/jittered

**Decision:** `scheduler-go`'s dispatcher retries a failed job with
`RetryBackoff * attempt` (linear) up to `MaxAttempts` (default 3), then
dead-letters it.

**Alternative considered:** Exponential backoff with jitter (the more
common production default).

**Why not:** For a synchronous-feeling async job queue where callers expect
results within seconds, exponential backoff's later retries (multi-second,
then tens of seconds) don't fit the workload — by the time a real retry
would fire, the caller has moved on. Linear backoff with a small fixed
window and a hard cap of 3 attempts keeps the failure-to-DLQ path bounded
and fast, which is what `dispatcher_test.go` actually verifies (3 calls,
then exactly one dead-lettered message). The tradeoff: this policy would be
too aggressive for a genuinely bursty failure (e.g. a worker restarting for
30s) — a production system would want backoff shaped by the failure mode,
which linear-with-a-cap deliberately doesn't attempt to distinguish.

## Promotion gate: accuracy delta threshold, not a full canary/shadow pipeline

**Decision:** `ml/pipelines/retrain_and_promote.py` promotes a candidate to
production only if its held-out accuracy beats the current production
version's by `--min-improvement` (default: any improvement); otherwise it
exits non-zero and leaves the candidate in `staging`.

**Alternative considered:** Shadow traffic / canary rollout with live
A-B comparison before promotion (the registry's `traffic_pct` field and
`SetTraffic` endpoint exist specifically to support this).

**Why not:** A held-out-accuracy gate is honest about what it verifies (the
model didn't regress on synthetic validation data) without claiming to
verify what it can't (real-world behavior under live traffic) — building a
convincing canary pipeline needs real traffic patterns this project doesn't
have. The registry already exposes the traffic-split primitive a canary
pipeline would need; wiring an actual shadow-traffic comparison on top of it
is named explicitly as a future improvement rather than faked with a
metrics-only gate dressed up as more than it is.
