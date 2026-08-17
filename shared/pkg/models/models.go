// Package models holds domain types shared across TeslaEdge services.
package models

import "time"

// Precision is the numeric precision an inference worker runs a model at.
type Precision string

const (
	PrecisionFP32 Precision = "fp32"
	PrecisionFP16 Precision = "fp16"
	PrecisionINT8 Precision = "int8"
	PrecisionINT4 Precision = "int4"
)

// Priority controls job-queue ordering.
type Priority int

const (
	PriorityNormal Priority = iota
	PriorityHigh
	PriorityCritical
)

// TelemetryEvent is a single sample emitted by a simulated vehicle.
type TelemetryEvent struct {
	VehicleID   string    `json:"vehicle_id"`
	Timestamp   time.Time `json:"timestamp"`
	Speed       float64   `json:"speed_kph"`
	Accel       float64   `json:"accel_ms2"`
	SteeringDeg float64   `json:"steering_deg"`
	Lat         float64   `json:"lat"`
	Lon         float64   `json:"lon"`
	EventLabel  string    `json:"event_label,omitempty"`
}

// InferenceJob is a unit of work dispatched by the scheduler to a worker.
type InferenceJob struct {
	ID              string    `json:"id"`
	VehicleID       string    `json:"vehicle_id"`
	ModelName       string    `json:"model_name"`
	ModelVersion    string    `json:"model_version"`
	PreferPrecision Precision `json:"preferred_precision"`
	Features        []float64 `json:"features"`
	Priority        Priority  `json:"priority"`
	SubmittedAt     time.Time `json:"submitted_at"`
	Attempt         int       `json:"attempt"`
	MaxAttempts     int       `json:"max_attempts"`
}

// InferenceResult is the outcome of running an InferenceJob.
type InferenceResult struct {
	JobID          string    `json:"job_id"`
	WorkerID       string    `json:"worker_id"`
	ModelName      string    `json:"model_name"`
	ModelVersion   string    `json:"model_version"`
	PrecisionUsed  Precision `json:"precision_used"`
	Output         []float64 `json:"output"`
	PredictedLabel string    `json:"predicted_label"`
	Confidence     float64   `json:"confidence"`
	LatencyMS      int64     `json:"latency_ms"`
	QueueTimeMS    int64     `json:"queue_time_ms"`
	Error          string    `json:"error,omitempty"`
}

// ModelStage is the lifecycle stage of a registered model version.
type ModelStage string

const (
	StageStaging    ModelStage = "staging"
	StageProduction ModelStage = "production"
	StageArchived   ModelStage = "archived"
)

// ModelVersion is one trained, registered artifact of a named model.
type ModelVersion struct {
	ModelName    string             `json:"model_name"`
	Version      string             `json:"version"`
	Stage        ModelStage         `json:"stage"`
	Precision    Precision          `json:"precision"`
	ArtifactPath string             `json:"artifact_path"`
	Metrics      map[string]float64 `json:"metrics"`
	TrafficPct   int                `json:"traffic_pct"`
	CreatedAt    time.Time          `json:"created_at"`
}

// WorkerInfo describes an inference worker registered with the router.
type WorkerInfo struct {
	ID              string    `json:"id"`
	SupportedModels []string  `json:"supported_models"`
	GPUAvailable    bool      `json:"gpu_available"`
	MaxPrecision    Precision `json:"max_precision"`
	LastLatencyMS   int64     `json:"last_latency_ms"`
	InFlight        int       `json:"in_flight"`
	LastHeartbeat   time.Time `json:"last_heartbeat"`
}
