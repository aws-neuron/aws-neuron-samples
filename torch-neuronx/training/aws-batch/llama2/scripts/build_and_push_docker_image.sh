#!/bin/bash
set -euo pipefail

export DOCKER_BUILDKIT=1

pushd ./docker
# Build a Neuron container image for running all-reduce test on AWS Batch and push the image to ECR
# Authenticate with ECR, build & push the image
aws ecr get-login-password --region $REGION | docker login --username AWS \
    --password-stdin $BASE_IMAGE_REPO \
  && docker build . -t aws-batch:latest \
    --build-arg BASE_IMAGE_REPO=$BASE_IMAGE_REPO \
    --build-arg BASE_IMAGE_NAME=$BASE_IMAGE_NAME \
    --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG

aws ecr get-login-password --region $REGION | docker login --username AWS \
    --password-stdin $ECR_REPO \
  && docker tag aws-batch:latest $ECR_REPO:latest \
  && docker push $ECR_REPO:latest
popd