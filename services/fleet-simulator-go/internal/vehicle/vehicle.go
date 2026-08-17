// Package vehicle simulates a single fleet vehicle: a goroutine that emits
// telemetry on a jittered interval and periodically fires an inference
// request against the gateway, mimicking an onboard driving-event classifier.
package vehicle

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"time"
)

// Config parameterizes a simulated vehicle.
type Config struct {
	ID              string
	GatewayURL      string
	APIKey          string
	TelemetryPeriod time.Duration
	InferEvery      int // send one inference request every N telemetry ticks
	HTTPClient      *http.Client
}

// Stats are counters a vehicle reports back to the simulator's aggregator.
type Stats struct {
	TelemetryTicks int64
	InferOK        int64
	InferErr       int64
}

type telemetryFrame struct {
	VehicleID   string  `json:"vehicle_id"`
	Speed       float64 `json:"speed_kph"`
	Accel       float64 `json:"accel_ms2"`
	SteeringDeg float64 `json:"steering_deg"`
	Lat         float64 `json:"lat"`
	Lon         float64 `json:"lon"`
}

type inferRequest struct {
	VehicleID string    `json:"vehicle_id"`
	ModelName string    `json:"model_name"`
	Precision string    `json:"precision"`
	Priority  string    `json:"priority"`
	Features  []float64 `json:"features"`
}

// Run drives one simulated vehicle until ctx is canceled, reporting Stats on
// each update to the statsCh channel (non-blocking best effort).
func Run(ctx context.Context, cfg Config, statsCh chan<- Stats) {
	rng := rand.New(rand.NewSource(hash(cfg.ID)))
	lat, lon := 37.4+rng.Float64()*0.5, -122.1-rng.Float64()*0.5

	ticker := time.NewTicker(jitter(rng, cfg.TelemetryPeriod))
	defer ticker.Stop()

	tick := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			tick++
			speed := 40 + 30*math.Sin(float64(tick)/10.0) + rng.NormFloat64()*3
			accel := rng.NormFloat64() * 1.5
			steering := rng.NormFloat64() * 5
			lat += rng.NormFloat64() * 0.0005
			lon += rng.NormFloat64() * 0.0005

			delta := Stats{TelemetryTicks: 1}

			if cfg.InferEvery > 0 && tick%cfg.InferEvery == 0 {
				features := []float64{speed, accel, steering, rng.NormFloat64(), rng.NormFloat64()}
				if err := sendInfer(ctx, cfg, features); err != nil {
					delta.InferErr = 1
				} else {
					delta.InferOK = 1
				}
			}

			_ = telemetryFrame{cfg.ID, speed, accel, steering, lat, lon}

			select {
			case statsCh <- delta:
			default:
			}
		}
	}
}

func sendInfer(ctx context.Context, cfg Config, features []float64) error {
	body, _ := json.Marshal(inferRequest{
		VehicleID: cfg.ID,
		ModelName: "driving-event-classifier",
		Precision: "fp32",
		Priority:  "normal",
		Features:  features,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.GatewayURL+"/v1/infer", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("X-API-Key", cfg.APIKey)
	}

	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("gateway returned %d", resp.StatusCode)
	}
	return nil
}

func jitter(rng *rand.Rand, base time.Duration) time.Duration {
	delta := time.Duration(rng.Int63n(int64(base) / 2))
	return base/2 + delta
}

func hash(s string) int64 {
	var h int64 = 1469598103934665603
	for _, c := range s {
		h ^= int64(c)
		h *= 1099511628211
	}
	if h < 0 {
		h = -h
	}
	return h
}
