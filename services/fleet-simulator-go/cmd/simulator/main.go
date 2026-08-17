// Command simulator runs TeslaEdge's fleet simulator: a configurable number
// of concurrent goroutines, each emulating one vehicle sending telemetry and
// occasional inference requests to the gateway.
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/fleet-simulator-go/internal/telemetry"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/fleet-simulator-go/internal/vehicle"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/config"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/logging"
)

func main() {
	log := logging.New("fleet-simulator")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fleetSize := config.Int("FLEET_SIZE", 100)
	gatewayURL := config.String("GATEWAY_URL", "http://localhost:8080")
	apiKey := config.String("GATEWAY_API_KEY", "")
	telemetryPeriod := config.Duration("TELEMETRY_PERIOD", time.Second)
	inferEvery := config.Int("INFER_EVERY_N_TICKS", 5)

	agg := telemetry.New()
	reg := prometheus.NewRegistry()
	agg.Register(reg)
	agg.ActiveVehicles.Set(float64(fleetSize))

	statsCh := make(chan vehicle.Stats, fleetSize*2)
	httpClient := &http.Client{Timeout: 5 * time.Second}

	var wg sync.WaitGroup
	for i := 0; i < fleetSize; i++ {
		cfg := vehicle.Config{
			ID:              vehicleID(i),
			GatewayURL:      gatewayURL,
			APIKey:          apiKey,
			TelemetryPeriod: telemetryPeriod,
			InferEvery:      inferEvery,
			HTTPClient:      httpClient,
		}
		wg.Add(1)
		go func(c vehicle.Config) {
			defer wg.Done()
			vehicle.Run(ctx, c, statsCh)
		}(cfg)
	}

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case delta := <-statsCh:
				agg.Observe(delta)
			}
		}
	}()

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	addr := config.String("SIMULATOR_METRICS_ADDR", ":9093")
	srv := &http.Server{Addr: addr, Handler: mux}

	go func() {
		log.Info("fleet simulator running", "fleet_size", fleetSize, "gateway", gatewayURL, "metrics_addr", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("metrics server stopped", "error", err)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down fleet simulator")
	wg.Wait()
}

func vehicleID(i int) string {
	return "vehicle-" + itoa(i)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
