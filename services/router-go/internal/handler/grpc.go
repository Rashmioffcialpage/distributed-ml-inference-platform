// Package handler implements the gRPC InferenceService for router-go: it
// selects a live worker via internal/routing and forwards the request to
// that worker's HTTP /predict endpoint, timing the round trip.
package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/router-go/internal/routing"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server implements inferencev1.InferenceServiceServer.
type Server struct {
	inferencev1.UnimplementedInferenceServiceServer

	Registry   *routing.Registry
	HTTPClient *http.Client
	Log        *slog.Logger
	Version    string
}

func precisionToDomain(p inferencev1.Precision) models.Precision {
	switch p {
	case inferencev1.Precision_PRECISION_FP32:
		return models.PrecisionFP32
	case inferencev1.Precision_PRECISION_FP16:
		return models.PrecisionFP16
	case inferencev1.Precision_PRECISION_INT8:
		return models.PrecisionINT8
	case inferencev1.Precision_PRECISION_INT4:
		return models.PrecisionINT4
	default:
		return ""
	}
}

func priorityToDomain(p inferencev1.Priority) models.Priority {
	switch p {
	case inferencev1.Priority_PRIORITY_HIGH:
		return models.PriorityHigh
	case inferencev1.Priority_PRIORITY_CRITICAL:
		return models.PriorityCritical
	default:
		return models.PriorityNormal
	}
}

type workerPredictRequest struct {
	Features  []float32 `json:"features"`
	Precision string    `json:"precision"`
	ModelName string    `json:"model_name"`
}

type workerPredictResponse struct {
	Output         []float32 `json:"output"`
	PredictedLabel string    `json:"predicted_label"`
	Confidence     float32   `json:"confidence"`
	ModelVersion   string    `json:"model_version"`
	PrecisionUsed  string    `json:"precision_used"`
}

// Predict selects the best live worker for the request and forwards it over HTTP.
func (s *Server) Predict(ctx context.Context, req *inferencev1.InferenceRequest) (*inferencev1.InferenceResponse, error) {
	start := time.Now()
	queueStart := req.SubmittedAtUnixMs
	var queueMS int64
	if queueStart > 0 {
		queueMS = start.UnixMilli() - queueStart
	}

	workers := s.Registry.Snapshot()
	w, err := routing.Select(workers, req.ModelName, precisionToDomain(req.PreferredPrecision), priorityToDomain(req.Priority))
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "route request: %v", err)
	}

	body, _ := json.Marshal(workerPredictRequest{
		Features:  req.Features,
		Precision: string(precisionToDomain(req.PreferredPrecision)),
		ModelName: req.ModelName,
	})

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, w.Endpoint+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "build worker request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "call worker %s: %v", w.ID, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, status.Errorf(codes.Internal, "worker %s returned %d: %s", w.ID, resp.StatusCode, string(b))
	}

	var wr workerPredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&wr); err != nil {
		return nil, status.Errorf(codes.Internal, "decode worker response: %v", err)
	}

	latency := time.Since(start).Milliseconds()
	s.Registry.Heartbeat(w.ID, latency, w.InFlight)

	return &inferencev1.InferenceResponse{
		RequestId:      req.RequestId,
		WorkerId:       w.ID,
		ModelName:      req.ModelName,
		ModelVersion:   wr.ModelVersion,
		PrecisionUsed:  req.PreferredPrecision,
		Output:         wr.Output,
		PredictedLabel: wr.PredictedLabel,
		Confidence:     wr.Confidence,
		LatencyMs:      latency,
		QueueTimeMs:    queueMS,
	}, nil
}

// StreamPredict pipelines many requests over one connection, reusing Predict per message.
func (s *Server) StreamPredict(stream inferencev1.InferenceService_StreamPredictServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		resp, err := s.Predict(stream.Context(), req)
		if err != nil {
			return err
		}
		if err := stream.Send(resp); err != nil {
			return err
		}
	}
}

// Health reports router + worker-pool status.
func (s *Server) Health(ctx context.Context, _ *inferencev1.HealthRequest) (*inferencev1.HealthResponse, error) {
	return &inferencev1.HealthResponse{
		Status:        "ok",
		ActiveWorkers: int32(s.Registry.Count()),
		RouterVersion: s.Version,
	}, nil
}

// RegisterWorker adds a worker to the live pool.
func (s *Server) RegisterWorker(ctx context.Context, req *inferencev1.RegisterWorkerRequest) (*inferencev1.RegisterWorkerResponse, error) {
	if req.WorkerId == "" || req.Endpoint == "" {
		return nil, status.Error(codes.InvalidArgument, "worker_id and endpoint are required")
	}
	s.Registry.Register(routing.Worker{
		WorkerInfo: models.WorkerInfo{
			ID:              req.WorkerId,
			SupportedModels: req.SupportedModels,
			GPUAvailable:    req.GpuAvailable,
			MaxPrecision:    precisionToDomain(req.MaxPrecision),
		},
		Endpoint: req.Endpoint,
	})
	s.Log.Info("worker registered", "worker_id", req.WorkerId, "endpoint", req.Endpoint, "models", req.SupportedModels)
	return &inferencev1.RegisterWorkerResponse{Accepted: true}, nil
}
