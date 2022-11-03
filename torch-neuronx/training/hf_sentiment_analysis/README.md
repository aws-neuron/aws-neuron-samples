# Hugging Face BERT Sentinment Analysis - AWS Trainium

## Introduction

In this example, we will go through the steps required for easily adapt your PyTorch code for training a Machine Learning
(ML) model by using [Hugging Face](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face) and BERT as 
model type on an Amazon EC2 instance by using AWS Trainium chip.

In this repository, we are sharing some code examples for:
1. Train BERT ML model by using PyTorch and Hugging Face on [CPU/GPU instances](./examples/00-cpu-gpu/train.py)
2. Train BERT ML model by using PyTorch and Hugging Face on a [single Neuron Core](./examples/01-trainium-single-core/train.py)
3. Distributed training of BERT ML model by using PyTorch and Hugging Face on [all the available Neuron Cores](./examples/02-trainium-distributed-training/train.py)

## Infrastructure Setup for AWS Trainium

For facilitating the setup of AWS Trainium-based Amazon EC2 instances, we are providing a [configuration script](./setup.sh) 
that can be executed in the Amazon EC2 instance.

### Prerequisites

* Instance Image: Amazon Linux AMI
* Instance Type: trn1.32xlarge
* Git installed on the EC2 instance

```
sudo yum update

sudo yum install git

git --version
```

### Run configuration setup

```
chmod +x setup.sh

./setup.sh
```

### Activate conda environment

```
source trainium-hg/bin/activate
```

### Check AWS Neuron SDK installation

```
export PATH=/opt/aws/neuron/bin/:$PATH

neuron-ls

neuron-top
```

## ML Training on single Neuron Core

Activate [conda environment](#activate-conda-environment) 

```
cd examples/01-trainium-single-core

python3 train.py
```

## Distributed Training on all available Neuron Cores

Activate [conda environment](#activate-conda-environment) 

```
cd examples/02-trainium-distributed-training

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=32 train.py
```

# Errors

1. Flush Neuron Cores

```
sudo rmmod neuron; sudo modprobe neuron
```