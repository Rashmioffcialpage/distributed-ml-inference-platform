variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name. Also used to derive resource names/tags."
  type        = string
  default     = "teslaedge"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS control plane."
  type        = string
  default     = "1.30"
}

variable "node_instance_type" {
  description = "EC2 instance type for the worker node group. t3.medium (2 vCPU / 4GB) is a cost-conscious default for a portfolio-scale deployment of this repo's 7 small service pods + 2 ML workers — bump this if the PyTorch ml-worker pods get OOMKilled under real load."
  type        = string
  default     = "t3.medium"
}

variable "node_desired_size" {
  description = "Desired worker node count."
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum worker node count (cluster autoscaler floor)."
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum worker node count (cluster autoscaler ceiling)."
  type        = number
  default     = 3
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC the cluster lives in."
  type        = string
  default     = "10.42.0.0/16"
}
