// Package queue defines a small backend-agnostic job queue interface used
// by the scheduler. TeslaEdge's roadmap calls for "Kafka / Redis" as the
// distributed messaging layer; rather than hard-wiring one, the scheduler
// programs against this interface and a concrete backend (Redis Streams by
// default, Kafka as a drop-in alternative, in-memory for unit tests) is
// selected at startup via the QUEUE_BACKEND env var.
package queue

import "context"

// Message is one enqueued unit of work. Payload is an opaque, backend-agnostic
// byte slice (JSON-encoded models.InferenceJob in practice).
type Message struct {
	ID      string
	Payload []byte
	// Attempt is how many times this message has been delivered, including
	// the current delivery. Backends that support redelivery (Redis Streams
	// consumer groups, Kafka with manual offset commit) populate this from
	// their own delivery-count tracking where available; otherwise it is
	// tracked by the caller in the payload itself.
	Attempt int
}

// Queue is a minimal durable work queue: enqueue, and consume with explicit
// ack/nack so failed jobs can be retried or dead-lettered by the caller.
type Queue interface {
	// Enqueue publishes a payload onto the named topic/stream.
	Enqueue(ctx context.Context, topic string, payload []byte) error
	// Consume delivers messages on the named topic to handler until ctx is
	// canceled. The handler must call Ack or Nack via the returned Message
	// semantics implicitly: returning nil acks, returning an error nacks
	// (the backend redelivers or the caller dead-letters after MaxAttempts).
	Consume(ctx context.Context, topic string, group string, handler func(context.Context, Message) error) error
	// DeadLetter publishes a payload that exhausted its retry budget onto a
	// topic's dead-letter counterpart (by convention "<topic>.dlq").
	DeadLetter(ctx context.Context, topic string, payload []byte, reason string) error
	Close() error
}
