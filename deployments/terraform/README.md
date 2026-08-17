# Terraform: AWS EKS + ECR for TeslaEdge

Provisions everything `docker compose up` doesn't cover for a real cloud
deployment: a VPC (via `terraform-aws-modules/vpc`), an EKS cluster with a
managed node group (via `terraform-aws-modules/eks`), and one ECR
repository per service image.

**This costs real money the moment you run `apply`** — an EKS control
plane alone is ~$0.10/hour (~$73/month) before any EC2 node or NAT gateway
cost. Full walkthrough, cost breakdown, and teardown instructions:
[../../docs/cloud-deploy.md](../../docs/cloud-deploy.md). Read that before
running anything here.

Quick reference, once you've read the walkthrough above:

```bash
cp terraform.tfvars.example terraform.tfvars   # edit as needed
terraform init
terraform plan
terraform apply
```

## A note on how this was validated

This Terraform was written and formatted (`terraform fmt`) in an
environment where `registry.terraform.io` is unreachable — so `terraform
init`/`plan`/`apply` were never run here, and neither the AWS credentials
present in that environment worked (`InvalidClientTokenId` on
`sts:GetCallerIdentity`). Both the `vpc` and `eks` module sources were
still fetched and inspected directly from their GitHub repos (`git clone`,
bypassing the registry), pinned to the exact tags referenced here
(`v5.9.0` / `v20.9.0`), and every variable and output name used in
`main.tf`/`outputs.tf` was cross-checked against that real source — but
that's static verification, not a real `plan`. Your `terraform init` here
will be the first time this configuration is actually exercised against
the Terraform Registry and a real AWS account.
