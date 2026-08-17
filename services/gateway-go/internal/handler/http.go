// Package handler implements the gateway's REST surface: synchronous
// inference (proxied to router-go over gRPC) and asynchronous job
// submission (enqueued for scheduler-go to process).
package handler

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/queue"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/tracing"
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
)

var tracer = tracing.Tracer("gateway")

// Handler holds the gateway's dependencies.
type Handler struct {
	Router      inferencev1.InferenceServiceClient
	Queue       queue.Queue
	Log         *slog.Logger
	JobsTopic   string
	CallTimeout time.Duration
}

type inferRequest struct {
	VehicleID    string    `json:"vehicle_id"`
	ModelName    string    `json:"model_name"`
	ModelVersion string    `json:"model_version"`
	Precision    string    `json:"precision"`
	Features     []float64 `json:"features"`
	Priority     string    `json:"priority"`
}

type inferResponse struct {
	RequestID      string  `json:"request_id"`
	WorkerID       string  `json:"worker_id"`
	ModelVersion   string  `json:"model_version"`
	PredictedLabel string  `json:"predicted_label"`
	Confidence     float32 `json:"confidence"`
	LatencyMS      int64   `json:"latency_ms"`
}

func precisionToProto(p string) inferencev1.Precision {
	switch models.Precision(p) {
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

func priorityToProto(p string) inferencev1.Priority {
	switch p {
	case "high":
		return inferencev1.Priority_PRIORITY_HIGH
	case "critical":
		return inferencev1.Priority_PRIORITY_CRITICAL
	default:
		return inferencev1.Priority_PRIORITY_NORMAL
	}
}

// Infer handles POST /v1/infer: a synchronous, low-latency inference call
// proxied straight through to the router.
func (h *Handler) Infer(w http.ResponseWriter, r *http.Request) {
	var req inferRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json body"}`, http.StatusBadRequest)
		return
	}
	if req.ModelName == "" || len(req.Features) == 0 {
		http.Error(w, `{"error":"model_name and features are required"}`, http.StatusBadRequest)
		return
	}

	features := make([]float32, len(req.Features))
	for i, f := range req.Features {
		features[i] = float32(f)
	}

	requestID := uuid.NewString()
	ctx, span := tracer.Start(r.Context(), "gateway.Infer")
	defer span.End()
	ctx, cancel := context.WithTimeout(ctx, h.CallTimeout)
	defer cancel()

	resp, err := h.Router.Predict(ctx, &inferencev1.InferenceRequest{
		RequestId:          requestID,
		VehicleId:          req.VehicleID,
		ModelName:          req.ModelName,
		ModelVersion:       req.ModelVersion,
		PreferredPrecision: precisionToProto(req.Precision),
		Features:           features,
		Priority:           priorityToProto(req.Priority),
		SubmittedAtUnixMs:  time.Now().UnixMilli(),
	})
	if err != nil {
		h.Log.Warn("inference call failed", "error", err, "request_id", requestID)
		http.Error(w, `{"error":"inference failed: `+err.Error()+`"}`, http.StatusBadGateway)
		return
	}

	writeJSON(w, http.StatusOK, inferResponse{
		RequestID:      resp.RequestId,
		WorkerID:       resp.WorkerId,
		ModelVersion:   resp.ModelVersion,
		PredictedLabel: resp.PredictedLabel,
		Confidence:     resp.Confidence,
		LatencyMS:      resp.LatencyMs,
	})
}

type jobAcceptedResponse struct {
	JobID string `json:"job_id"`
}

// SubmitJob handles POST /v1/jobs: enqueues an asynchronous inference job
// for scheduler-go to dispatch with retry/backoff semantics, instead of
// blocking the caller on a synchronous round trip.
func (h *Handler) SubmitJob(w http.ResponseWriter, r *http.Request) {
	var req inferRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json body"}`, http.StatusBadRequest)
		return
	}
	if req.ModelName == "" || len(req.Features) == 0 {
		http.Error(w, `{"error":"model_name and features are required"}`, http.StatusBadRequest)
		return
	}

	job := models.InferenceJob{
		ID:              uuid.NewString(),
		VehicleID:       req.VehicleID,
		ModelName:       req.ModelName,
		ModelVersion:    req.ModelVersion,
		PreferPrecision: models.Precision(req.Precision),
		Features:        req.Features,
		Priority:        priorityFromString(req.Priority),
		SubmittedAt:     time.Now(),
		MaxAttempts:     3,
	}
	payload, _ := json.Marshal(job)

	if err := h.Queue.Enqueue(r.Context(), h.JobsTopic, payload); err != nil {
		h.Log.Error("enqueue job failed", "error", err)
		http.Error(w, `{"error":"failed to enqueue job"}`, http.StatusServiceUnavailable)
		return
	}

	writeJSON(w, http.StatusAccepted, jobAcceptedResponse{JobID: job.ID})
}

func priorityFromString(p string) models.Priority {
	switch p {
	case "high":
		return models.PriorityHigh
	case "critical":
		return models.PriorityCritical
	default:
		return models.PriorityNormal
	}
}

// Healthz handles GET /healthz.
func (h *Handler) Healthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
