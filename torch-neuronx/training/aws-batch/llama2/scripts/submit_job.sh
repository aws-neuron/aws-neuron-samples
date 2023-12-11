#!/bin/bash
set -eu

# submitting aws batch job
aws batch submit-job \
    --job-name $JOB_NAME \
    --job-queue $JOB_QUEUE_NAME \
    --job-definition $JOB_DEF_NAME \
    --node-overrides numNodes=4
