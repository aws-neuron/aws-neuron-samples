# AWS Batch / trn1 allreduce example

This package shows how to run a basic multi-node allreduce test using trn1.32xlarge instances in AWS Batch. A successful allreduce test indicates that the Neuron driver, Neuron SDK, and EFA driver are installed properly, and the required EFA device configuration + connectivity is in place to support multi-node training.

It is expected that these scripts will be run from an x86_64-based Linux instance.

Note: to use trn1n.32xlarge instances, the launch template and job definition will need to be adjusted to use 16 EFA devices (currently using 8 EFA devices for trn1.32xlarge).

Prereqs:
* Existing VPC with subnet and appropriate [EFA security group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-security)
* ECR repo
* AWS CLI installed and configured with permissions for Batch and ECR
* Docker installed 
* jq installed

Steps:
* Modify `build_configs_and_scripts.sh` with your account/region/etc
* Run `./build_configs_and_scripts.sh` to create the configs/scripts using your config details
* Run `./create_resources.sh` to create the various AWS Batch resources (job definition, compute environment, ...)
* Run `./build_docker_image.sh` to build a basic Neuron-capable container image using the latest Neuron packages and push the image to ECR
* Run `./submit_job.sh` to submit a basic 4-node allreduce job in the provisioned Batch environment