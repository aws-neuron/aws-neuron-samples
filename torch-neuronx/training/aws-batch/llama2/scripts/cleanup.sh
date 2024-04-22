#!/bin/bash
set -euo pipefail

aws ec2 delete-placement-group --group-name $PLACEMENT_GROUP_NAME                                                           # deleting the placement group
aws ec2 delete-launch-template --launch-template-name $LAUNCH_TEMPLATE_NAME                                                 # deleting the aws batch compute environment

aws batch update-job-queue --job-queue $JOB_QUEUE_NAME --state DISABLED                                                     # disabling the job queue
while [[ ! $( aws batch describe-job-queues --job-queue $JOB_QUEUE_NAME | jq -r ".jobQueues[].state") =~ DISABLED ]]
do
        echo -n "."
        sleep 2
done
aws batch delete-job-queue --job-queue $JOB_QUEUE_NAME                                                                      # deleting the job queue
while [[ $(aws batch describe-job-queues --job-queue $JOB_QUEUE_NAME | jq -r '.jobQueues | length') -ne 0 ]]; do
    echo -n "."
    sleep 5
done

aws batch update-compute-environment --compute-environment $COMPUTE_ENV_NAME --state DISABLED                               # disabling the compute environment
while [[ ! $(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME  | jq -r ".computeEnvironments[].status") =~ VALID ]]
do
        echo -n "."
        sleep 5
done
aws batch delete-compute-environment --compute-environment $COMPUTE_ENV_NAME                                                # deleting the compute environment
aws batch deregister-job-definition --job-definition $JOB_DEF_NAME                                                          # deregistering the aws batch job definition
echo -e "\nCleaned up all the resources."