// Package client wraps the gRPC connection from gateway-go to router-go.
package client

import (
	inferencev1 "github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/proto/inference/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// NewRouterClient dials router-go's gRPC endpoint (insecure: both services
// run inside the same trusted network in docker-compose/k8s; put this
// behind mTLS before exposing it beyond the cluster).
func NewRouterClient(addr string) (inferencev1.InferenceServiceClient, func() error, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, err
	}
	return inferencev1.NewInferenceServiceClient(conn), conn.Close, nil
}
