#!/bin/bash
set -eu

# submitting aws batch job
aws batch submit-job \
    --job-name aws-batch-trn1-job \
    --job-queue aws-batch-job-queue \
    --job-definition aws-batch-job-definition \
    --node-overrides numNodes=4