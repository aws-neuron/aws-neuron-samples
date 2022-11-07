# Hugging Face BERT Sentinment Analysis - AWS Trainium

## Introduction

In this example, we will go through the steps required for easily adapt your PyTorch code for training a Machine Learning
(ML) model by using [Hugging Face](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face) and BERT as 
model type on an Amazon EC2 instance by using AWS Trainium chip.

In this repository, we are sharing some code examples for:
1. Train BERT ML model by using PyTorch and Hugging Face
   1. Code: [single Neuron Core](code/01-trainium-single-core/train.py)
   2. Notebook: [notebook single Neuron Core](./01-hf-single-neuron.ipynb)
2. Distributed training of BERT ML model by using PyTorch and Hugging Face
   1. Code: [distributed training on Neuron Cores](code/02-trainium-distributed-training/train.py)
   2. Notebook: [notebook distributed training on Neuron Cores](./02-hf-distributed-training.ipynb)

## Infrastructure Setup for AWS Trainium

### Prerequisites

* Instance Image: [Deep Learning AMI Neuron PyTorch 1.11](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-pytorch-1-11-amazon-linux-2/)
* Instance Type: trn1.32xlarge
* Git installed on the EC2 instance

```
git --version
```

### Activate pre-built pytorch environment

```
source /opt/aws_neuron_venv_pytorch/bin/activate
```

### Check AWS Neuron SDK installation

```
neuron-ls

neuron-top
```

## ML Training on single Neuron Core

Activate [pre-built pytorch environment](#activate-pre-built-pytorch-environment) 

Test the code execution by using the provided [notebook](./01-hf-single-neuron.ipynb)

### CL execution example

```
cd examples/01-trainium-single-core

python3 train.py
```

## Distributed Training on all available Neuron Cores

Activate [pre-built pytorch environment](#activate-pre-built-pytorch-environment) 

Test the code execution by using the provided [notebook](./02-hf-distributed-training.ipynb)

### CL execution example

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