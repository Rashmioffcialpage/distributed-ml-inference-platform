// Command scheduler runs TeslaEdge's job scheduler: it consumes inference
// jobs from the queue (Redis Streams or Kafka), dispatches them to
// router-go, and retries/dead-letters failures.
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/scheduler-go/internal/worker"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/config"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/logging"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/queue"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/tracing"
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	log := logging.New("scheduler")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownTracing, err := tracing.Init(ctx, "scheduler-go")
	if err != nil {
		log.Warn("tracing init failed, continuing without spans", "error", err)
	} else {
		defer shutdownTracing(context.Background())
	}

	routerAddr := config.String("ROUTER_GRPC_ADDR", "localhost:9090")
	conn, err := grpc.NewClient(routerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error("failed to dial router", "addr", routerAddr, "error", err)
		os.Exit(1)
	}
	defer conn.Close()

	q := queue.FromEnv()
	defer q.Close()

	processed := promauto.NewCounter(prometheus.CounterOpts{Namespace: "scheduler", Name: "jobs_processed_total"})
	failed := promauto.NewCounter(prometheus.CounterOpts{Namespace: "scheduler", Name: "jobs_failed_total"})
	retried := promauto.NewCounter(prometheus.CounterOpts{Namespace: "scheduler", Name: "jobs_retried_total"})
	deadLettered := promauto.NewCounter(prometheus.CounterOpts{Namespace: "scheduler", Name: "jobs_dead_lettered_total"})

	d := &worker.Dispatcher{
		Queue:        q,
		Router:       inferencev1.NewInferenceServiceClient(conn),
		Log:          log,
		JobsTopic:    config.String("JOBS_TOPIC", "inference.jobs"),
		ResultsTopic: config.String("RESULTS_TOPIC", "inference.results"),
		CallTimeout:  config.Duration("ROUTER_CALL_TIMEOUT", 5*time.Second),
		RetryBackoff: config.Duration("RETRY_BACKOFF", 200*time.Millisecond),
		Metrics: &worker.Metrics{
			Processed:  processed.Inc,
			Failed:     failed.Inc,
			Retried:    retried.Inc,
			DeadLetter: deadLettered.Inc,
		},
	}

	go func() {
		log.Info("scheduler dispatch loop starting", "jobs_topic", d.JobsTopic, "router", routerAddr)
		if err := d.Run(ctx); err != nil {
			log.Error("dispatcher stopped", "error", err)
		}
	}()

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	addr := config.String("SCHEDULER_METRICS_ADDR", ":9092")
	srv := &http.Server{Addr: addr, Handler: mux}

	go func() {
		log.Info("scheduler metrics listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("metrics server stopped", "error", err)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down scheduler")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}
