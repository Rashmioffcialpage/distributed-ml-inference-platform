package routing

import (
	"fmt"
	"math"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
)

// precisionRank orders precisions from cheapest/fastest to most expensive so
// we can tell whether a worker's max precision satisfies a request.
var precisionRank = map[models.Precision]int{
	models.PrecisionINT4: 0,
	models.PrecisionINT8: 1,
	models.PrecisionFP16: 2,
	models.PrecisionFP32: 3,
}

// Select implements the routing policy described in the roadmap: pick a
// worker based on model support, GPU availability, current latency/load and
// workload priority. It returns an error if no eligible worker is live.
//
// Scoring (lower is better):
//
//	score = latencyMS + inFlight*loadWeight - gpuBonus(if priority == Critical)
//
// Workers that don't support the requested model, or whose max precision
// can't satisfy the request, are filtered out entirely before scoring.
func Select(workers []*Worker, modelName string, preferred models.Precision, priority models.Priority) (*Worker, error) {
	var eligible []*Worker
	for _, w := range workers {
		if !supportsModel(w, modelName) {
			continue
		}
		if preferred != "" && precisionRank[w.MaxPrecision] < precisionRank[preferred] {
			continue
		}
		eligible = append(eligible, w)
	}
	if len(eligible) == 0 {
		return nil, fmt.Errorf("no eligible worker for model %q precision %q (of %d live workers)", modelName, preferred, len(workers))
	}

	const loadWeight = 15.0 // ms penalty per in-flight request
	const gpuBonus = 25.0   // ms discount for GPU workers on critical priority

	best := eligible[0]
	bestScore := math.MaxFloat64
	for _, w := range eligible {
		score := float64(w.LastLatencyMS) + float64(w.InFlight)*loadWeight
		if priority == models.PriorityCritical && w.GPUAvailable {
			score -= gpuBonus
		}
		if score < bestScore {
			bestScore = score
			best = w
		}
	}
	return best, nil
}

func supportsModel(w *Worker, modelName string) bool {
	for _, m := range w.SupportedModels {
		if m == modelName {
			return true
		}
	}
	return false
}
