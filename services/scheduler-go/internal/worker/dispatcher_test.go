package worker

import (
	"context"
	"encoding/json"
	"log/slog"
	"net"
	"testing"
	"time"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/queue"
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// fakeRouter always fails Predict, to exercise the dispatcher's retry/DLQ path.
type fakeRouter struct {
	inferencev1.UnimplementedInferenceServiceServer
	calls int
}

func (f *fakeRouter) Predict(ctx context.Context, req *inferencev1.InferenceRequest) (*inferencev1.InferenceResponse, error) {
	f.calls++
	return nil, context.DeadlineExceeded
}

func dialFakeRouter(t *testing.T, fr *fakeRouter) inferencev1.InferenceServiceClient {
	t.Helper()
	lis := bufconn.Listen(1024 * 1024)
	s := grpc.NewServer()
	inferencev1.RegisterInferenceServiceServer(s, fr)
	go s.Serve(lis)
	t.Cleanup(s.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("dial bufconn: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return inferencev1.NewInferenceServiceClient(conn)
}

func TestDispatcher_DeadLettersAfterMaxAttempts(t *testing.T) {
	fr := &fakeRouter{}
	q := queue.NewMemory()
	d := &Dispatcher{
		Queue:        q,
		Router:       dialFakeRouter(t, fr),
		Log:          slog.Default(),
		JobsTopic:    "jobs",
		ResultsTopic: "results",
		CallTimeout:  time.Second,
		RetryBackoff: time.Millisecond,
	}

	job := models.InferenceJob{ID: "job-1", ModelName: "event-classifier", MaxAttempts: 3}
	payload, _ := json.Marshal(job)

	ctx, cancel := context.WithCancel(context.Background())
	go d.Run(ctx)

	if err := q.Enqueue(context.Background(), "jobs", payload); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	deadline := time.After(2 * time.Second)
	for {
		if dl := q.DeadLettered("jobs"); len(dl) == 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("job was never dead-lettered after %d router calls", fr.calls)
		case <-time.After(10 * time.Millisecond):
		}
	}
	cancel()

	if fr.calls != 3 {
		t.Fatalf("want 3 attempts (MaxAttempts), got %d", fr.calls)
	}
}
