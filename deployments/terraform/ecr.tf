# One ECR repository per image this repo builds. Matches the 6 Dockerfiles
# under deployments/docker/ — ml-worker is a single image shared by both the
# fp32 and int8 Deployments (see deployments/kubernetes/30-ml-workers.yaml),
# so it doesn't need two repos.
locals {
  services = [
    "router",
    "gateway",
    "scheduler",
    "model-deployer",
    "fleet-simulator",
    "ml-worker",
  ]
}

resource "aws_ecr_repository" "service" {
  for_each             = toset(local.services)
  name                 = "teslaedge/${each.value}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Untagged images (superseded pushes to the same tag, e.g. repeated
# `:latest` pushes during iteration) expire after 7 days instead of
# accumulating storage cost forever.
resource "aws_ecr_lifecycle_policy" "service" {
  for_each   = aws_ecr_repository.service
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 7 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = { type = "expire" }
    }]
  })
}
