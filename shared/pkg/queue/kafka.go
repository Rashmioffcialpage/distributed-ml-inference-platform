package queue

import (
	"context"
	"strings"

	kafka "github.com/segmentio/kafka-go"
)

// Kafka implements Queue on top of segmentio/kafka-go. It exists to show
// TeslaEdge's messaging layer is not tied to Redis: swap QUEUE_BACKEND=kafka
// and point KAFKA_BROKERS at a cluster to get partitioned, replayable topics
// with consumer-group semantics instead of Redis Streams.
type Kafka struct {
	brokers []string
	writers map[string]*kafka.Writer
}

// NewKafka returns a Queue backed by the given comma-separated broker list.
func NewKafka(brokers string) *Kafka {
	return &Kafka{
		brokers: strings.Split(brokers, ","),
		writers: make(map[string]*kafka.Writer),
	}
}

func (k *Kafka) writerFor(topic string) *kafka.Writer {
	if w, ok := k.writers[topic]; ok {
		return w
	}
	w := &kafka.Writer{
		Addr:     kafka.TCP(k.brokers...),
		Topic:    topic,
		Balancer: &kafka.LeastBytes{},
	}
	k.writers[topic] = w
	return w
}

func (k *Kafka) Enqueue(ctx context.Context, topic string, payload []byte) error {
	return k.writerFor(topic).WriteMessages(ctx, kafka.Message{Value: payload})
}

func (k *Kafka) Consume(ctx context.Context, topic string, group string, handler func(context.Context, Message) error) error {
	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers: k.brokers,
		Topic:   topic,
		GroupID: group,
	})
	defer r.Close()

	for {
		m, err := r.FetchMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			continue
		}
		msg := Message{ID: string(m.Key), Payload: m.Value, Attempt: 1}
		if err := handler(ctx, msg); err != nil {
			continue // not committed; redelivered on next fetch after rebalance
		}
		_ = r.CommitMessages(ctx, m)
	}
}

func (k *Kafka) DeadLetter(ctx context.Context, topic string, payload []byte, reason string) error {
	w := k.writerFor(topic + ".dlq")
	return w.WriteMessages(ctx, kafka.Message{Value: payload, Headers: []kafka.Header{{Key: "reason", Value: []byte(reason)}}})
}

func (k *Kafka) Close() error {
	for _, w := range k.writers {
		_ = w.Close()
	}
	return nil
}
