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

### Prerequisites

* Instance Image: Deep Learning AMI Neuron PyTorch 1.11
* Instance Type: trn1.32xlarge
* Git installed on the EC2 instance

```
git --version
```

### Activate conda environment

```
source /opt/aws_neuron_venv_pytorch/bin/activate
```

### Check AWS Neuron SDK installation

```
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