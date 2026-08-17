// Package routing implements TeslaEdge's inference-router core: a thread
// safe registry of live inference workers plus the selection algorithm that
// picks which worker should serve a given request.
package routing

import (
	"sort"
	"sync"
	"time"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
)

// Worker is a registered inference worker, as tracked by the router. It
// extends models.WorkerInfo with the HTTP endpoint the router forwards
// Predict calls to.
type Worker struct {
	models.WorkerInfo
	Endpoint string
}

// Registry holds the live worker pool. Workers expire (are considered dead)
// if they don't heartbeat within staleAfter.
type Registry struct {
	mu         sync.RWMutex
	workers    map[string]*Worker
	staleAfter time.Duration
	rrCounter  int
}

// NewRegistry returns an empty worker registry.
func NewRegistry(staleAfter time.Duration) *Registry {
	return &Registry{workers: make(map[string]*Worker), staleAfter: staleAfter}
}

// Register adds or refreshes a worker's registration.
func (r *Registry) Register(w Worker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	w.LastHeartbeat = time.Now()
	r.workers[w.ID] = &w
}

// Heartbeat marks a worker alive and updates its last observed latency/load.
func (r *Registry) Heartbeat(id string, latencyMS int64, inFlight int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if w, ok := r.workers[id]; ok {
		w.LastHeartbeat = time.Now()
		w.LastLatencyMS = latencyMS
		w.InFlight = inFlight
	}
}

// Snapshot returns the currently live (non-stale) workers.
func (r *Registry) Snapshot() []*Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	cutoff := time.Now().Add(-r.staleAfter)
	out := make([]*Worker, 0, len(r.workers))
	for _, w := range r.workers {
		if w.LastHeartbeat.After(cutoff) {
			out = append(out, w)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Count returns the number of currently live workers.
func (r *Registry) Count() int {
	return len(r.Snapshot())
}
