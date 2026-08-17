// Package config provides minimal environment-variable configuration loading
// shared by every TeslaEdge service. It intentionally avoids a heavyweight
// config framework: each service declares the handful of env vars it needs.
package config

import (
	"os"
	"strconv"
	"time"
)

// String returns the environment variable value or a fallback default.
func String(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

// Int returns the environment variable parsed as an int, or a fallback default.
func Int(key string, def int) int {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

// Bool returns the environment variable parsed as a bool, or a fallback default.
func Bool(key string, def bool) bool {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return def
	}
	return b
}

// Duration returns the environment variable parsed as a time.Duration, or a fallback default.
func Duration(key string, def time.Duration) time.Duration {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return def
	}
	return d
}
