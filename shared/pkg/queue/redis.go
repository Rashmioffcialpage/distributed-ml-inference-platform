package queue

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

// Redis implements Queue using Redis Streams + consumer groups, giving us
// at-least-once delivery, per-consumer-group offsets and automatic
// redelivery of unacked messages (via XPENDING/XCLAIM in a production
// hardening pass) with none of Kafka's operational overhead — the default
// backend for local dev and the docker-compose stack.
type Redis struct {
	client *redis.Client
}

// NewRedis connects to a Redis server at addr (e.g. "redis:6379").
func NewRedis(addr string) *Redis {
	return &Redis{client: redis.NewClient(&redis.Options{Addr: addr})}
}

func (r *Redis) Enqueue(ctx context.Context, topic string, payload []byte) error {
	return r.client.XAdd(ctx, &redis.XAddArgs{
		Stream: topic,
		Values: map[string]interface{}{"payload": payload},
	}).Err()
}

func (r *Redis) Consume(ctx context.Context, topic string, group string, handler func(context.Context, Message) error) error {
	// Best-effort group creation; ignore "already exists".
	_ = r.client.XGroupCreateMkStream(ctx, topic, group, "0").Err()

	consumer := group + "-consumer"
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		streams, err := r.client.XReadGroup(ctx, &redis.XReadGroupArgs{
			Group:    group,
			Consumer: consumer,
			Streams:  []string{topic, ">"},
			Count:    10,
			Block:    2 * time.Second,
		}).Result()
		if err != nil {
			if err == redis.Nil || ctx.Err() != nil {
				continue
			}
			continue
		}

		for _, s := range streams {
			for _, xmsg := range s.Messages {
				payload, _ := xmsg.Values["payload"].(string)
				msg := Message{ID: xmsg.ID, Payload: []byte(payload), Attempt: 1}
				if err := handler(ctx, msg); err != nil {
					continue // left pending; a redelivery sweep can XCLAIM it
				}
				r.client.XAck(ctx, topic, group, xmsg.ID)
			}
		}
	}
}

func (r *Redis) DeadLetter(ctx context.Context, topic string, payload []byte, reason string) error {
	return r.client.XAdd(ctx, &redis.XAddArgs{
		Stream: topic + ".dlq",
		Values: map[string]interface{}{"payload": payload, "reason": reason},
	}).Err()
}

func (r *Redis) Close() error {
	return r.client.Close()
}
