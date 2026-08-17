package routing

import (
	"testing"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
)

func worker(id string, gpu bool, precision models.Precision, latency int64, inFlight int, supports ...string) *Worker {
	return &Worker{WorkerInfo: models.WorkerInfo{
		ID: id, GPUAvailable: gpu, MaxPrecision: precision,
		LastLatencyMS: latency, InFlight: inFlight, SupportedModels: supports,
	}}
}

func TestSelect_PrefersLowerLatency(t *testing.T) {
	workers := []*Worker{
		worker("slow", false, models.PrecisionFP32, 200, 0, "event-classifier"),
		worker("fast", false, models.PrecisionFP32, 20, 0, "event-classifier"),
	}
	got, err := Select(workers, "event-classifier", "", models.PriorityNormal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != "fast" {
		t.Fatalf("want fast worker, got %s", got.ID)
	}
}

func TestSelect_FiltersUnsupportedModel(t *testing.T) {
	workers := []*Worker{
		worker("w1", false, models.PrecisionFP32, 10, 0, "trajectory"),
	}
	_, err := Select(workers, "event-classifier", "", models.PriorityNormal)
	if err == nil {
		t.Fatal("expected error when no worker supports the model")
	}
}

func TestSelect_FiltersInsufficientPrecision(t *testing.T) {
	workers := []*Worker{
		worker("int8-only", false, models.PrecisionINT8, 5, 0, "event-classifier"),
	}
	_, err := Select(workers, "event-classifier", models.PrecisionFP32, models.PriorityNormal)
	if err == nil {
		t.Fatal("expected error: worker maxes out at int8, request wants fp32")
	}
}

func TestSelect_CriticalPriorityPrefersGPU(t *testing.T) {
	workers := []*Worker{
		worker("cpu", false, models.PrecisionFP32, 30, 0, "event-classifier"),
		worker("gpu", true, models.PrecisionFP32, 40, 0, "event-classifier"),
	}
	got, err := Select(workers, "event-classifier", "", models.PriorityCritical)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != "gpu" {
		t.Fatalf("want gpu worker for critical priority, got %s", got.ID)
	}
}

func TestSelect_LoadPenalizesBusyWorker(t *testing.T) {
	workers := []*Worker{
		worker("busy", false, models.PrecisionFP32, 10, 20, "event-classifier"),
		worker("idle", false, models.PrecisionFP32, 15, 0, "event-classifier"),
	}
	got, err := Select(workers, "event-classifier", "", models.PriorityNormal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != "idle" {
		t.Fatalf("want idle worker despite slightly higher base latency, got %s", got.ID)
	}
}

func BenchmarkSelect(b *testing.B) {
	workers := make([]*Worker, 0, 200)
	for i := 0; i < 200; i++ {
		workers = append(workers, worker("w", i%3 == 0, models.PrecisionFP32, int64(10+i), i%5, "event-classifier"))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Select(workers, "event-classifier", "", models.PriorityNormal)
	}
}
