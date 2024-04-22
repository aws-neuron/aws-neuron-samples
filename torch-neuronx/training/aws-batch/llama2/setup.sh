#!/bin/bash
set -euo pipefail

# Read variables from config file
source config.txt

export REGION
export SUBNET
export SG
export ECR_REPO
export INSTANCE_ROLE
export DO_PRE_COMPILATION
export TOKENIZED_DATASET_URI
export NEURON_COMPILE_CACHE_URI
export CHECKPOINT_SAVE_URI

# ECR repo and image details. You can locate the correct Neuron DLC image for 'training' on AWS DLC github page - https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers
export BASE_IMAGE_REPO=763104351884.dkr.ecr.$REGION.amazonaws.com
export BASE_IMAGE_NAME=pytorch-training-neuronx
export BASE_IMAGE_TAG=1.13.1-neuronx-py310-sdk2.18.0-ubuntu20.04
export ECS_AMI_NAME=/aws/service/ecs/optimized-ami/amazon-linux-2023/recommended/image_id
export ECS_AMI=$(aws ssm get-parameter --region $REGION --name $ECS_AMI_NAME | jq -r .Parameter.Value)

export PLACEMENT_GROUP_NAME=aws-batch-placement-group
export LAUNCH_TEMPLATE_NAME=aws-batch-launch-template
export COMPUTE_ENV_NAME=aws-batch-compute-environment
export JOB_QUEUE_NAME=aws-batch-job-queue
export JOB_DEF_NAME=aws-batch-job-definition
export JOB_NAME=aws-batch-job

export USER_DATA=$(cat << EOF | base64 -w0
"MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/cloud-boothook; charset="us-ascii"

#!/bin/bash
sudo yum install -y libibverbs-utils rdma-core-devel ibacm infiniband-diags-compat librdmacm-utils
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

# Creating directories required for setup
mkdir -p ./data
mkdir -p ./build
mkdir -p ./docker/llama2

# Locating and moving tokenizer to required directory
if [[ ! -e "tokenizer.model" ]]; then
  echo "Tokenizer File does not exist. Please ensure you have tokenizer file placed in the root directory with the name as 'tokenizer.model'"
  exit 1
fi
mv tokenizer.model ./data/

# Downloading the sample files required for data pre-processing
wget -q -P ./data/ https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/get_dataset.py
wget -q -P ./data/ https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/config.json

# Environment substitution is required files
for template in ./templates/*.json; do envsubst < $template > ./build/`basename $template`; done
for script in ./scripts/*.sh; do envsubst < $script > ./`basename $script`; chmod u+x ./`basename $script`; done
envsubst  '$DO_PRE_COMPILATION $NEURON_COMPILE_CACHE_URI $CHECKPOINT_SAVE_URI $TOKENIZED_DATASET_URI' < ./docker/llama_batch_training.sh > ./docker/llama2/llama_batch_training.sh

# Downloading the sample files required for Llama training
pushd . > /dev/null
cd ./docker/llama2
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/tp_zero1_llama2_7b_hf_pretrain.py
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/logger.py
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/modeling_llama_nxd.py
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/requirements.txt
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/config.json
wget -q https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/training_utils.py
popd > /dev/null
echo "Set up has been completed successfully."