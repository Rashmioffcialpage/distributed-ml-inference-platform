// Command deployer runs TeslaEdge's model-deployer service: the model
// registry API for registering versions, deploying, rolling back and
// splitting canary traffic, backed by Postgres.
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/model-deployer-go/internal/db"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/model-deployer-go/internal/handler"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/model-deployer-go/internal/registry"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/config"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/logging"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/metrics"
)

func main() {
	log := logging.New("model-deployer")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	dsn := config.String("POSTGRES_DSN", "postgres://teslaedge:teslaedge@localhost:5432/teslaedge?sslmode=disable")
	pool, err := db.Connect(ctx, dsn)
	if err != nil {
		log.Error("failed to connect to postgres", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	m := metrics.New("model_deployer")
	h := &handler.Handler{Registry: registry.New(pool), Log: log}

	mux := http.NewServeMux()
	route := func(pattern string, fn http.HandlerFunc) {
		mux.HandleFunc(pattern, m.Middleware(pattern, fn))
	}
	route("POST /v1/models/register", h.Register)
	route("GET /v1/models/{name}/versions", h.List)
	route("GET /v1/models/{name}/production", h.Production)
	route("POST /v1/models/{name}/deploy", h.Deploy)
	route("POST /v1/models/{name}/rollback", h.Rollback)
	route("POST /v1/models/{name}/traffic", h.Traffic)
	mux.HandleFunc("GET /healthz", h.Healthz)
	mux.Handle("GET /metrics", promhttp.Handler())

	addr := config.String("DEPLOYER_ADDR", ":8081")
	srv := &http.Server{Addr: addr, Handler: mux}

	go func() {
		log.Info("model-deployer listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server stopped", "error", err)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down model-deployer")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}
