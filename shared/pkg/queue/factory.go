package queue

import "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/config"

// FromEnv builds a Queue from QUEUE_BACKEND ("memory" | "redis" | "kafka"),
// defaulting to "redis" since that's what docker-compose wires up.
func FromEnv() Queue {
	switch config.String("QUEUE_BACKEND", "redis") {
	case "kafka":
		return NewKafka(config.String("KAFKA_BROKERS", "localhost:9092"))
	case "memory":
		return NewMemory()
	default:
		return NewRedis(config.String("REDIS_ADDR", "localhost:6379"))
	}
}
