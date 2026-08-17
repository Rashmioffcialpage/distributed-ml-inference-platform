package queue

import (
	"context"
	"sync"
)

// Memory is an in-process Queue implementation used for unit tests and for
// running the scheduler standalone without Redis/Kafka.
type Memory struct {
	mu     sync.Mutex
	topics map[string]chan Message
	dlq    map[string][][]byte
	seq    int
	closed bool
}

// NewMemory returns a ready-to-use in-memory Queue.
func NewMemory() *Memory {
	return &Memory{
		topics: make(map[string]chan Message),
		dlq:    make(map[string][][]byte),
	}
}

func (m *Memory) chanFor(topic string) chan Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	ch, ok := m.topics[topic]
	if !ok {
		ch = make(chan Message, 1024)
		m.topics[topic] = ch
	}
	return ch
}

func (m *Memory) Enqueue(ctx context.Context, topic string, payload []byte) error {
	m.mu.Lock()
	m.seq++
	id := itoa(m.seq)
	m.mu.Unlock()

	select {
	case m.chanFor(topic) <- Message{ID: id, Payload: payload, Attempt: 1}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Memory) Consume(ctx context.Context, topic string, _ string, handler func(context.Context, Message) error) error {
	ch := m.chanFor(topic)
	for {
		select {
		case <-ctx.Done():
			return nil
		case msg := <-ch:
			if err := handler(ctx, msg); err != nil {
				msg.Attempt++
				select {
				case ch <- msg:
				default:
				}
			}
		}
	}
}

func (m *Memory) DeadLetter(_ context.Context, topic string, payload []byte, _ string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dlq[topic] = append(m.dlq[topic], payload)
	return nil
}

// DeadLettered returns the payloads dead-lettered for a topic (test helper).
func (m *Memory) DeadLettered(topic string) [][]byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([][]byte(nil), m.dlq[topic]...)
}

func (m *Memory) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
