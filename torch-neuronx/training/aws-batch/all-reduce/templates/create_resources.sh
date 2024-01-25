#!/bin/bash
set -eu

if [ ! `which jq` ]
then
	echo "Please install jq and re-run this script" && exit
fi

aws ec2 create-placement-group --group-name "aws-batch-placement-group" --strategy "cluster" --region $REGION   # creating the aws placement group
aws ec2 create-launch-template --cli-input-json file://build/launch_template.json              # creating the aws launch template
aws batch create-compute-environment --cli-input-json file://build/compute_env.json            # creating the aws batch compute environment
aws batch register-job-definition --cli-input-json file://build/job_def.json                   # creating the aws batch job definition
while [[ ! $(aws batch describe-compute-environments --compute-environments aws-batch-compute-environment | jq -r ".computeEnvironments[].status") =~ VALID ]]
do
        echo -n "."
        sleep 2
done
aws batch create-job-queue --cli-input-json file://build/job_queue.json                        # creating the aws batch job queue