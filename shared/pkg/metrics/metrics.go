// Package metrics defines the Prometheus metrics common to every TeslaEdge
// Go service (request rate, latency, in-flight, queue depth, errors) plus a
// small net/http middleware that records them for REST handlers.
package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Set is the common metric set every service registers. Services may add
// their own metrics on top of this (e.g. queue depth in the scheduler).
type Set struct {
	RequestsTotal   *prometheus.CounterVec
	RequestDuration *prometheus.HistogramVec
	InFlight        prometheus.Gauge
	ErrorsTotal     *prometheus.CounterVec
}

// New registers and returns the common metric set for a service, namespaced
// so metrics from different services don't collide (e.g. "gateway_requests_total").
func New(service string) *Set {
	return &Set{
		RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: service,
			Name:      "requests_total",
			Help:      "Total number of requests handled.",
		}, []string{"route", "method", "status"}),
		RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: service,
			Name:      "request_duration_seconds",
			Help:      "Request duration in seconds.",
			Buckets:   []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		}, []string{"route", "method"}),
		InFlight: promauto.NewGauge(prometheus.GaugeOpts{
			Namespace: service,
			Name:      "in_flight_requests",
			Help:      "Number of requests currently being processed.",
		}),
		ErrorsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: service,
			Name:      "errors_total",
			Help:      "Total number of handler errors.",
		}, []string{"route", "kind"}),
	}
}

// Middleware wraps an http.Handler, recording request count and latency.
func (s *Set) Middleware(route string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.InFlight.Inc()
		defer s.InFlight.Dec()

		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next(rec, r)

		s.RequestDuration.WithLabelValues(route, r.Method).Observe(time.Since(start).Seconds())
		s.RequestsTotal.WithLabelValues(route, r.Method, strconv.Itoa(rec.status)).Inc()
		if rec.status >= 500 {
			s.ErrorsTotal.WithLabelValues(route, "server_error").Inc()
		}
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}
