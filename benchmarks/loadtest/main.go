// Command loadtest is TeslaEdge's load-test tool: N concurrent goroutines
// hammer the gateway's POST /v1/infer endpoint for a fixed duration and
// report throughput, p50/p95/p99 latency, and error rate — the "load-test
// results" the README's portfolio quality bar calls for, measured for
// real rather than invented.
//
// Usage:
//
//	go run ./benchmarks/loadtest -url http://localhost:8080 -concurrency 50 -duration 20s -api-key devkey
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type inferRequest struct {
	VehicleID string    `json:"vehicle_id"`
	ModelName string    `json:"model_name"`
	Precision string    `json:"precision"`
	Priority  string    `json:"priority"`
	Features  []float64 `json:"features"`
}

type result struct {
	latency time.Duration
	ok      bool
}

func main() {
	url := flag.String("url", "http://localhost:8080", "gateway base URL")
	concurrency := flag.Int("concurrency", 50, "number of concurrent workers")
	duration := flag.Duration("duration", 20*time.Second, "how long to run")
	apiKey := flag.String("api-key", "devkey", "X-API-Key header value")
	modelName := flag.String("model", "driving-event-classifier", "model_name to request")
	flag.Parse()

	client := &http.Client{Timeout: 5 * time.Second}
	resultsCh := make(chan result, 100000)
	stop := make(chan struct{})

	var sent int64
	var wg sync.WaitGroup
	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(workerID)))
			for {
				select {
				case <-stop:
					return
				default:
				}
				body, _ := json.Marshal(inferRequest{
					VehicleID: fmt.Sprintf("loadtest-%d", workerID),
					ModelName: *modelName,
					Precision: "fp32",
					Priority:  "normal",
					Features:  []float64{rng.Float64() * 120, rng.NormFloat64() * 3, rng.NormFloat64() * 10, rng.NormFloat64(), rng.NormFloat64()},
				})

				start := time.Now()
				req, _ := http.NewRequest(http.MethodPost, *url+"/v1/infer", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				if *apiKey != "" {
					req.Header.Set("X-API-Key", *apiKey)
				}
				resp, err := client.Do(req)
				lat := time.Since(start)
				atomic.AddInt64(&sent, 1)

				ok := err == nil && resp != nil && resp.StatusCode == http.StatusOK
				if resp != nil {
					resp.Body.Close()
				}
				resultsCh <- result{latency: lat, ok: ok}
			}
		}(i)
	}

	go func() {
		time.Sleep(*duration)
		close(stop)
	}()

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	var latencies []time.Duration
	var okCount, errCount int
	for r := range resultsCh {
		latencies = append(latencies, r.latency)
		if r.ok {
			okCount++
		} else {
			errCount++
		}
	}

	if len(latencies) == 0 {
		fmt.Fprintln(os.Stderr, "no requests completed")
		os.Exit(1)
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	total := len(latencies)
	rps := float64(total) / duration.Seconds()

	fmt.Printf("TeslaEdge load test: %s  concurrency=%d  duration=%s\n", *url, *concurrency, *duration)
	fmt.Printf("total requests:   %d\n", total)
	fmt.Printf("successful:       %d (%.2f%%)\n", okCount, 100*float64(okCount)/float64(total))
	fmt.Printf("failed:           %d (%.2f%%)\n", errCount, 100*float64(errCount)/float64(total))
	fmt.Printf("throughput:       %.1f req/s\n", rps)
	fmt.Printf("p50 latency:      %s\n", percentile(latencies, 50))
	fmt.Printf("p95 latency:      %s\n", percentile(latencies, 95))
	fmt.Printf("p99 latency:      %s\n", percentile(latencies, 99))
	fmt.Printf("max latency:      %s\n", latencies[total-1])
}

func percentile(sorted []time.Duration, p int) time.Duration {
	idx := (p * len(sorted)) / 100
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
