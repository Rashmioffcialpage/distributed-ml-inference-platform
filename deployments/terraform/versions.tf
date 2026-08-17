terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state by default so `terraform init` works with zero prerequisites.
  # For anything beyond a solo portfolio deploy, switch to a remote backend
  # (S3 + DynamoDB lock table) before the first `apply` — local state has no
  # locking and is easy to lose. See docs/cloud-deploy.md.
  #
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "teslaedge/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "your-terraform-lock-table"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
}
