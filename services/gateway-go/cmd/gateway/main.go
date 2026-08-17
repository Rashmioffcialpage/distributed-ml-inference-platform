// Command gateway runs TeslaEdge's API Gateway: the REST entrypoint for
// vehicle/simulator traffic, handling auth, rate limiting, health checks and
// proxying inference requests to router-go.
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/gateway-go/internal/auth"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/gateway-go/internal/client"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/gateway-go/internal/handler"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/gateway-go/internal/ratelimit"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/config"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/logging"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/metrics"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/queue"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/tracing"
)

func main() {
	log := logging.New("gateway")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownTracing, err := tracing.Init(ctx, "gateway-go")
	if err != nil {
		log.Warn("tracing init failed, continuing without spans", "error", err)
	} else {
		defer shutdownTracing(context.Background())
	}

	routerAddr := config.String("ROUTER_GRPC_ADDR", "localhost:9090")
	routerClient, closeRouter, err := client.NewRouterClient(routerAddr)
	if err != nil {
		log.Error("failed to dial router", "addr", routerAddr, "error", err)
		os.Exit(1)
	}
	defer closeRouter()

	q := queue.FromEnv()
	defer q.Close()

	keys := auth.NewKeySet(splitCSV(config.String("GATEWAY_API_KEYS", "")))
	limiter := ratelimit.New(float64(config.Int("RATE_LIMIT_RPS", 50)), config.Int("RATE_LIMIT_BURST", 100))

	m := metrics.New("gateway")
	h := &handler.Handler{
		Router:      routerClient,
		Queue:       q,
		Log:         log,
		JobsTopic:   config.String("JOBS_TOPIC", "inference.jobs"),
		CallTimeout: config.Duration("ROUTER_CALL_TIMEOUT", 5*time.Second),
	}

	mux := http.NewServeMux()
	chain := func(route string, fn http.HandlerFunc) http.HandlerFunc {
		return m.Middleware(route, keys.Middleware(limiter.Middleware(fn)))
	}
	mux.HandleFunc("POST /v1/infer", chain("/v1/infer", h.Infer))
	mux.HandleFunc("POST /v1/jobs", chain("/v1/jobs", h.SubmitJob))
	mux.HandleFunc("GET /healthz", h.Healthz)
	mux.Handle("GET /metrics", promhttp.Handler())

	addr := config.String("GATEWAY_ADDR", ":8080")
	srv := &http.Server{Addr: addr, Handler: mux}

	go func() {
		log.Info("gateway listening", "addr", addr, "router", routerAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("gateway server stopped", "error", err)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down gateway")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
