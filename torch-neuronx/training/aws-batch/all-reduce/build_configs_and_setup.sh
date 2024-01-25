#!/bin/bash
set -eu

# ECR repo and image details. You can locate the correct Neuron DLC image for 'training' on AWS DLC github page - https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers
export BASE_IMAGE_REPO=763104351884.dkr.ecr.us-west-2.amazonaws.com
export BASE_IMAGE_NAME=pytorch-training-neuronx
export BASE_IMAGE_TAG=1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04

# Configure your account specific settings below
export REGION=<your-region-name>
export ACCOUNT=<your-account-name>
export INSTANCE_ROLE=<your-iam-instance-role-name>
export SUBNET=<your-subnet-name>
export SG=<your-secutrity-group-name>
export ECR_REPO=<your-ecr-repo-name>

ECS_AMI_NAME=/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id
export ECS_AMI=$(aws ssm get-parameter --region $REGION --name $ECS_AMI_NAME | jq -r .Parameter.Value)
export USERDATA=$(cat << EOF | base64 -w0
"MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/cloud-boothook; charset="us-ascii"

cloud-init-per once yum_wget yum install -y wget
cloud-init-per once wget_efa wget -q --timeout=20 https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-latest.tar.gz -O /tmp/aws-efa-installer-latest.tar.gz
cloud-init-per once tar_efa tar -xf /tmp/aws-efa-installer-latest.tar.gz -C /tmp
pushd /tmp/aws-efa-installer
cloud-init-per once install_efa ./efa_installer.sh -y
pop /tmp/aws-efa-installer

cloud-init-per once efa_info /opt/amazon/efa/bin/fi_info -p efa

cloud-init-per once neuron_driver1 echo -e "[neuron]\nname=Neuron YUM Repository\nbaseurl=https://yum.repos.neuron.amazonaws.com\nenabled=1\nmetadata_expire=0" | tee /etc/yum.repos.d/neuron.repo > /dev/null
cloud-init-per once neuron_driver2 rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
cloud-init-per once neuron_driver3 yum update -y
cloud-init-per once neuron_driver4 yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y
cloud-init-per once neuron_driver5 yum erase aws-neuronx-dkms -y
cloud-init-per once neuron_driver6 yum install aws-neuronx-dkms-2.* -y

--==MYBOUNDARY==--"
EOF
)

# Apply variable substitutions to template files and resource creation script
mkdir -p ./build

for i in ./templates/*.json; do
  echo $i -\> ./build/`basename $i`;
  envsubst < $i > ./build/`basename $i`;
done

envsubst < ./templates/create_resources.sh > ./create_resources.sh \
    && chmod u+x ./create_resources.sh \
    && echo ./templates/create_resources.sh -\> ./create_resources.sh
envsubst < ./templates/build_docker_image.sh > ./build_docker_image.sh \
    && chmod u+x ./build_docker_image.sh \
    && echo ./templates/build_docker_image.sh -\> ./build_docker_image.sh