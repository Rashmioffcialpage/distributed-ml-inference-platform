# Cloud deploy: TeslaEdge on AWS EKS

A step-by-step path from an empty AWS account to a running TeslaEdge
cluster, using `deployments/terraform` (VPC + EKS + ECR) and
`deployments/kubernetes/overlays/aws` (the same base manifests
`docker compose up`'s local dev flow doesn't touch, retargeted at ECR).

**This was written and validated as far as it's possible to without a
working AWS account or reachable Terraform Registry** — see
[What was and wasn't validated](#what-was-and-wasnt-validated) at the
bottom before trusting any of it blindly. Nothing here has been
`terraform apply`'d for real.

## This costs real money

Before running anything below, understand the ongoing cost once `terraform
apply` succeeds:

| Resource | Approx. cost |
|---|---|
| EKS control plane | $0.10/hour (~$73/month) |
| 2× `t3.medium` nodes (on-demand) | ~$0.0832/hour each (~$121/month total) |
| NAT gateway | ~$0.045/hour + data processing (~$33/month) |
| ECR storage | ~$0.10/GB-month (small for these images) |
| **Total, running continuously** | **~$230/month** |

None of this is "free tier forever." If you're deploying this to try it
out rather than to run it long-term, **do the teardown step
([below](#teardown-do-this-when-youre-done)) the same session** — an EKS
cluster left running over a weekend by accident is a real, avoidable
charge on a real credit card.

## Prerequisites

- An AWS account with billing enabled, and credentials for an IAM
  identity that can create VPCs, EKS clusters, EC2 instances, and ECR
  repositories (`AdministratorAccess` is the simple path for a portfolio
  deploy; scope it down if this is a shared account).
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html),
  configured (`aws configure` or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
  env vars) — verify with `aws sts get-caller-identity`.
- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5.
- `kubectl`.
- [`kustomize`](https://kubectl.docs.kubernetes.io/installation/kustomize/)
  (standalone binary — `kubectl`'s built-in `-k` support can *build*
  kustomizations but not `kustomize edit`, which
  `build-and-push-ecr.sh` needs). `go install sigs.k8s.io/kustomize/kustomize/v5@latest`
  works if you have Go.
- Docker, logged in enough to build images locally.

## 1. Provision the cluster

```bash
cd deployments/terraform
cp terraform.tfvars.example terraform.tfvars   # edit region/instance size/etc. if you want
terraform init
terraform plan     # review what it's about to create
terraform apply    # takes ~12-15 minutes, mostly waiting on the EKS control plane
```

`apply` prints outputs including `ecr_repository_urls` and
`configure_kubectl`. Point `kubectl` at the new cluster:

```bash
$(terraform output -raw configure_kubectl)
kubectl get nodes   # should show 2 Ready nodes within a few minutes
```

## 2. Build and push images

From the repo root:

```bash
./deployments/scripts/build-and-push-ecr.sh
```

This builds all 6 images (5 Go services + the shared ml-worker image),
pushes each to the ECR repo Terraform created, tags with both the current
git short SHA and `:latest`, and rewrites
`deployments/kubernetes/overlays/aws/kustomization.yaml`'s `images:` list
to point at what it just pushed — you don't need to hand-edit that file
unless you'd rather not run the script.

## 3. Deploy

```bash
kubectl apply -k deployments/kubernetes/overlays/aws
kubectl -n teslaedge get pods -w   # watch until everything is Running
```

This applies the same namespace/configmap/Redis/Postgres/five-Go-services/
two-ML-workers topology `docker compose up` runs locally — see
`deployments/kubernetes/base/` for the actual manifests, none of which
changed for the cloud path except image references.

## 4. Reach it

The gateway is a `ClusterIP` Service by default (matches local dev — no
public endpoint until you decide to expose one):

```bash
kubectl -n teslaedge port-forward svc/gateway 8080:8080
curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" -H "X-API-Key: devkey" \
  -d '{"vehicle_id":"vehicle-1","model_name":"driving-event-classifier","precision":"fp32","priority":"normal","features":[80,-6.5,25,0.1,0.2]}'
```

`deployments/kubernetes/base/40-ingress.yaml` is included but not applied
by default — it assumes an ingress-nginx controller is already installed
and a `teslaedge.local` DNS entry, which doesn't make sense as a default
for a fresh cloud cluster. For a real public endpoint on EKS, the more
idiomatic path is the
[AWS Load Balancer Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/)
provisioning a real ALB from an Ingress resource — not set up here; a
reasonable next step, not something to fake as already done.

## Teardown (do this when you're done)

```bash
kubectl delete -k deployments/kubernetes/overlays/aws   # so LoadBalancer/PV resources release cleanly first
cd deployments/terraform
terraform destroy
```

Confirm in the AWS Console (EC2, EKS, VPC) that nothing's left running —
`terraform destroy` is generally thorough, but it's your money, worth the
30 seconds to check.

## What was and wasn't validated

Written in a sandbox where `registry.terraform.io` returns `403 Forbidden`
at the discovery-document level — the same host serves both provider and
module metadata, so there was no path to a real `terraform init`/`plan`
here, and the AWS credentials present in that sandbox didn't work either
(`InvalidClientTokenId` on `sts:GetCallerIdentity`). What *was* actually
done, not just written and hoped:

- `terraform fmt -check` passes.
- Both the `vpc` and `eks` module sources were fetched directly from their
  GitHub repos (`git clone`, bypassing the blocked registry) at the exact
  tags this config references (`v5.9.0` / `v20.9.0`), and every variable
  and output name used in `main.tf`/`outputs.tf` was cross-checked against
  that real source.
- The kustomize overlay (`deployments/kubernetes/overlays/aws`) was
  actually built with a real `kustomize` binary — caught and fixed a
  cycle-detection error from the overlay initially being nested inside the
  base directory (now restructured as sibling `base/`/`overlays/`
  directories, the standard kustomize layout).
- `build-and-push-ecr.sh` was dry-run end to end against stubbed
  `aws`/`docker`/`terraform` binaries standing in for the real CLIs,
  confirmed to build each of the 6 images with the correct
  `SERVICE`/`CMD_PATH` build args, push the correct tags, and correctly
  rewrite the overlay's `kustomization.yaml` via `kustomize edit set image`
  — verified by rebuilding the overlay afterward and checking the image
  references actually changed.

What wasn't, and can't be from here: an actual `terraform apply` against a
real AWS account, and everything downstream of that (real EKS cluster
coming up healthy, real image pushes succeeding, pods actually scheduling
and passing health checks on real nodes). Your `terraform apply` in step 1
is the first time this whole path runs for real — expect to debug small
things (IAM permission gaps, a region without your chosen instance type,
etc.) the way you would with any first real run of infra-as-code.
