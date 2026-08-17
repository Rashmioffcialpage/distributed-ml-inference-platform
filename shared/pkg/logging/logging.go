// Package logging configures a structured slog.Logger used consistently
// across every TeslaEdge service so log lines are trivially parseable by
// Grafana Loki / any JSON-aware log pipeline.
package logging

import (
	"log/slog"
	"os"
)

// New returns a JSON structured logger tagged with the given service name.
func New(service string) *slog.Logger {
	level := slog.LevelInfo
	if os.Getenv("LOG_LEVEL") == "debug" {
		level = slog.LevelDebug
	}
	h := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})
	return slog.New(h).With("service", service)
}
