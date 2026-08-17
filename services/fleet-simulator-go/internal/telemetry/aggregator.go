// Package telemetry aggregates per-vehicle Stats into fleet-wide Prometheus
// gauges so a single Grafana panel can show total fleet traffic.
package telemetry

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/fleet-simulator-go/internal/vehicle"
)

// Aggregator sums Stats across all simulated vehicles.
type Aggregator struct {
	mu    sync.Mutex
	total vehicle.Stats

	TelemetryTotal prometheus.Counter
	InferOKTotal   prometheus.Counter
	InferErrTotal  prometheus.Counter
	ActiveVehicles prometheus.Gauge
}

// New registers the aggregator's Prometheus metrics.
func New() *Aggregator {
	return &Aggregator{
		TelemetryTotal: prometheus.NewCounter(prometheus.CounterOpts{Namespace: "fleet_simulator", Name: "telemetry_ticks_total"}),
		InferOKTotal:   prometheus.NewCounter(prometheus.CounterOpts{Namespace: "fleet_simulator", Name: "infer_ok_total"}),
		InferErrTotal:  prometheus.NewCounter(prometheus.CounterOpts{Namespace: "fleet_simulator", Name: "infer_err_total"}),
		ActiveVehicles: prometheus.NewGauge(prometheus.GaugeOpts{Namespace: "fleet_simulator", Name: "active_vehicles"}),
	}
}

// Register adds all metrics to the given registerer.
func (a *Aggregator) Register(reg prometheus.Registerer) {
	reg.MustRegister(a.TelemetryTotal, a.InferOKTotal, a.InferErrTotal, a.ActiveVehicles)
}

// Observe folds one vehicle's per-tick Stats delta into the aggregator's
// Prometheus counters.
func (a *Aggregator) Observe(delta vehicle.Stats) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if delta.TelemetryTicks > 0 {
		a.TelemetryTotal.Add(float64(delta.TelemetryTicks))
	}
	if delta.InferOK > 0 {
		a.InferOKTotal.Add(float64(delta.InferOK))
	}
	if delta.InferErr > 0 {
		a.InferErrTotal.Add(float64(delta.InferErr))
	}
}
