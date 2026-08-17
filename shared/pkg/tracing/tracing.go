// Package tracing configures OpenTelemetry distributed tracing shared by
// every TeslaEdge service. Spans propagate across the Gateway -> Router ->
// Scheduler -> Inference Worker call chain via the standard OTel HTTP/gRPC
// propagators, so a single trace ID follows one inference request end to end.
//
// By default spans are exported to stdout (zero external dependencies, easy
// to inspect locally with `docker compose logs`). Set OTEL_EXPORTER_OTLP_ENDPOINT
// to ship spans to a real collector (Jaeger/Tempo/etc.) in a deployed
// environment instead.
package tracing

import (
	"context"
	"os"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
)

// Init configures the global OTel tracer provider for a service and returns
// a shutdown function the caller should defer.
func Init(ctx context.Context, service string) (func(context.Context) error, error) {
	// Deliberately schemaless (not resource.Default()): the installed SDK's
	// default resource and the semconv package version this code imports
	// drift independently across otel releases, and merging two different
	// schema URLs is a hard error. A service name is all this project needs.
	res := resource.NewSchemaless(semconv.ServiceName(service))

	var sp sdktrace.SpanExporter
	var err error
	if endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"); endpoint != "" {
		sp, err = otlptracegrpc.New(ctx, otlptracegrpc.WithEndpoint(endpoint), otlptracegrpc.WithInsecure())
	} else {
		sp, err = stdouttrace.New(stdouttrace.WithoutTimestamps())
	}
	if err != nil {
		return nil, err
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(sp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{},
	))
	return tp.Shutdown, nil
}

// Tracer returns a named tracer for a service, e.g. Tracer("gateway").
func Tracer(name string) trace.Tracer {
	return otel.Tracer(name)
}

// GRPCServerOption instruments a gRPC server so every RPC it handles
// continues (or starts) a trace, using the global tracer provider Init set up.
func GRPCServerOption() grpc.ServerOption {
	return grpc.StatsHandler(otelgrpc.NewServerHandler())
}

// GRPCClientOption instruments a gRPC client connection so every RPC it
// makes carries the caller's trace context to the server (via the
// standard W3C traceparent propagator), letting Gateway -> Router and
// Scheduler -> Router calls join a single trace.
func GRPCClientOption() grpc.DialOption {
	return grpc.WithStatsHandler(otelgrpc.NewClientHandler())
}
