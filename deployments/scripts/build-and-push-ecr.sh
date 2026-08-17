#!/usr/bin/env bash
# Builds every TeslaEdge image and pushes it to the ECR repositories that
# deployments/terraform provisions, then rewrites the AWS kustomize
# overlay's image references to point at what was just pushed.
#
# Prerequisites (see docs/cloud-deploy.md):
#   - `terraform apply` already run in deployments/terraform, so the ECR
#     repos exist.
#   - AWS CLI configured with credentials that can push to ECR
#     (`aws sts get-caller-identity` should succeed).
#   - kustomize CLI installed (`go install sigs.k8s.io/kustomize/kustomize/v5@latest`
#     or see https://kubectl.docs.kubernetes.io/installation/kustomize/).
#
# Usage: run from the repo root.
#   ./deployments/scripts/build-and-push-ecr.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TF_DIR="deployments/terraform"
OVERLAY_DIR="deployments/kubernetes/overlays/aws"
TAG="$(git rev-parse --short HEAD)"

command -v aws >/dev/null || { echo "aws CLI not found — install it first." >&2; exit 1; }
command -v kustomize >/dev/null || { echo "kustomize CLI not found — install it first (see script header)." >&2; exit 1; }

AWS_REGION="$(terraform -chdir="$TF_DIR" output -raw aws_region 2>/dev/null || echo "${AWS_REGION:-us-east-1}")"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Logging in to ${REGISTRY}"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"

declare -A GO_SERVICES=(
  [router]="router-go:./cmd/router"
  [gateway]="gateway-go:./cmd/gateway"
  [scheduler]="scheduler-go:./cmd/scheduler"
  [model-deployer]="model-deployer-go:./cmd/deployer"
  [fleet-simulator]="fleet-simulator-go:./cmd/simulator"
)

for name in "${!GO_SERVICES[@]}"; do
  IFS=":" read -r service cmd_path <<<"${GO_SERVICES[$name]}"
  image="${REGISTRY}/teslaedge/${name}"
  echo "==> Building ${name}"
  docker build \
    -f deployments/docker/Dockerfile.go-service \
    --build-arg SERVICE="$service" \
    --build-arg CMD_PATH="$cmd_path" \
    -t "${image}:${TAG}" -t "${image}:latest" .
  echo "==> Pushing ${name}"
  docker push "${image}:${TAG}"
  docker push "${image}:latest"
  (cd "$OVERLAY_DIR" && kustomize edit set image "teslaedge/${name}=${image}:${TAG}")
done

echo "==> Building ml-worker"
image="${REGISTRY}/teslaedge/ml-worker"
docker build -f deployments/docker/Dockerfile.ml-worker -t "${image}:${TAG}" -t "${image}:latest" .
echo "==> Pushing ml-worker"
docker push "${image}:${TAG}"
docker push "${image}:latest"
(cd "$OVERLAY_DIR" && kustomize edit set image "teslaedge/ml-worker=${image}:${TAG}")

echo
echo "All images pushed and tagged :${TAG}. ${OVERLAY_DIR}/kustomization.yaml updated."
echo "Next: kubectl apply -k ${OVERLAY_DIR}"
