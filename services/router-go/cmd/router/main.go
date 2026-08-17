// Command router runs TeslaEdge's inference router: a gRPC service that
// selects the best live inference worker for each request and forwards it.
package main

import (
	"context"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/router-go/internal/handler"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/router-go/internal/routing"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/config"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/logging"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/tracing"
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

const version = "0.1.0"

func main() {
	log := logging.New("router")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownTracing, err := tracing.Init(ctx, "router-go")
	if err != nil {
		log.Warn("tracing init failed, continuing without spans", "error", err)
	} else {
		defer shutdownTracing(context.Background())
	}

	grpcAddr := config.String("ROUTER_GRPC_ADDR", ":9090")
	metricsAddr := config.String("ROUTER_METRICS_ADDR", ":9091")
	staleAfter := config.Duration("WORKER_STALE_AFTER", 30*time.Second)

	registry := routing.NewRegistry(staleAfter)

	srv := &handler.Server{
		Registry:   registry,
		HTTPClient: &http.Client{Timeout: config.Duration("WORKER_CALL_TIMEOUT", 5*time.Second)},
		Log:        log,
		Version:    version,
	}

	grpcServer := grpc.NewServer()
	inferencev1.RegisterInferenceServiceServer(grpcServer, srv)
	healthSrv := health.NewServer()
	healthpb.RegisterHealthServer(grpcServer, healthSrv)
	reflection.Register(grpcServer)

	lis, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		log.Error("failed to listen", "addr", grpcAddr, "error", err)
		os.Exit(1)
	}

	go func() {
		log.Info("router gRPC listening", "addr", grpcAddr)
		if err := grpcServer.Serve(lis); err != nil {
			log.Error("grpc server stopped", "error", err)
		}
	}()

	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsMux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok","active_workers":` + itoa(registry.Count()) + `}`))
	})
	metricsServer := &http.Server{Addr: metricsAddr, Handler: metricsMux}
	go func() {
		log.Info("router metrics listening", "addr", metricsAddr)
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("metrics server stopped", "error", err)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down router")
	grpcServer.GracefulStop()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = metricsServer.Shutdown(shutdownCtx)
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
