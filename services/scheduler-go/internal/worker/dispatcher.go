// Package worker implements scheduler-go's job dispatch loop: consume
// InferenceJob messages from the queue, route them through router-go, and
// handle retries/backoff/dead-lettering on failure.
package worker

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/queue"
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
)

// Dispatcher pulls jobs off a queue topic and executes them against the router.
type Dispatcher struct {
	Queue        queue.Queue
	Router       inferencev1.InferenceServiceClient
	Log          *slog.Logger
	JobsTopic    string
	ResultsTopic string
	CallTimeout  time.Duration
	RetryBackoff time.Duration

	Metrics *Metrics
}

// Metrics are the counters the scheduler exposes via Prometheus.
type Metrics struct {
	Processed  func()
	Failed     func()
	Retried    func()
	DeadLetter func()
}

func precisionToProto(p models.Precision) inferencev1.Precision {
	switch p {
	case models.PrecisionFP32:
		return inferencev1.Precision_PRECISION_FP32
	case models.PrecisionFP16:
		return inferencev1.Precision_PRECISION_FP16
	case models.PrecisionINT8:
		return inferencev1.Precision_PRECISION_INT8
	case models.PrecisionINT4:
		return inferencev1.Precision_PRECISION_INT4
	default:
		return inferencev1.Precision_PRECISION_UNSPECIFIED
	}
}

func priorityToProto(p models.Priority) inferencev1.Priority {
	switch p {
	case models.PriorityHigh:
		return inferencev1.Priority_PRIORITY_HIGH
	case models.PriorityCritical:
		return inferencev1.Priority_PRIORITY_CRITICAL
	default:
		return inferencev1.Priority_PRIORITY_NORMAL
	}
}

// Run consumes jobs until ctx is canceled.
func (d *Dispatcher) Run(ctx context.Context) error {
	return d.Queue.Consume(ctx, d.JobsTopic, "scheduler", func(ctx context.Context, msg queue.Message) error {
		var job models.InferenceJob
		if err := json.Unmarshal(msg.Payload, &job); err != nil {
			d.Log.Error("dropping malformed job", "error", err)
			return nil // ack; retrying a malformed payload never succeeds
		}
		return d.handle(ctx, job, msg.Payload)
	})
}

func (d *Dispatcher) handle(ctx context.Context, job models.InferenceJob, raw []byte) error {
	job.Attempt++
	features := make([]float32, len(job.Features))
	for i, f := range job.Features {
		features[i] = float32(f)
	}

	callCtx, cancel := context.WithTimeout(ctx, d.CallTimeout)
	defer cancel()

	resp, err := d.Router.Predict(callCtx, &inferencev1.InferenceRequest{
		RequestId:          job.ID,
		VehicleId:          job.VehicleID,
		ModelName:          job.ModelName,
		ModelVersion:       job.ModelVersion,
		PreferredPrecision: precisionToProto(job.PreferPrecision),
		Features:           features,
		Priority:           priorityToProto(job.Priority),
		SubmittedAtUnixMs:  job.SubmittedAt.UnixMilli(),
	})
	if err != nil {
		d.Log.Warn("job attempt failed", "job_id", job.ID, "attempt", job.Attempt, "error", err)
		if d.Metrics != nil {
			d.Metrics.Failed()
		}

		if job.Attempt >= job.MaxAttempts {
			if d.Metrics != nil {
				d.Metrics.DeadLetter()
			}
			return d.Queue.DeadLetter(ctx, d.JobsTopic, raw, err.Error())
		}

		if d.Metrics != nil {
			d.Metrics.Retried()
		}
		time.Sleep(d.RetryBackoff * time.Duration(job.Attempt)) // linear backoff
		updated, _ := json.Marshal(job)
		if enqueueErr := d.Queue.Enqueue(ctx, d.JobsTopic, updated); enqueueErr != nil {
			d.Log.Error("failed to requeue job", "job_id", job.ID, "error", enqueueErr)
		}
		return nil // ack the original delivery; the retry is a new message
	}

	if d.Metrics != nil {
		d.Metrics.Processed()
	}

	result := models.InferenceResult{
		JobID:          job.ID,
		WorkerID:       resp.WorkerId,
		ModelName:      resp.ModelName,
		ModelVersion:   resp.ModelVersion,
		PredictedLabel: resp.PredictedLabel,
		Confidence:     float64(resp.Confidence),
		LatencyMS:      resp.LatencyMs,
		QueueTimeMS:    resp.QueueTimeMs,
	}
	payload, _ := json.Marshal(result)
	if err := d.Queue.Enqueue(ctx, d.ResultsTopic, payload); err != nil {
		d.Log.Error("failed to publish result", "job_id", job.ID, "error", err)
	}
	return nil
}
