# Distributed Data Parallel (DDP) Examples

This folder contains examples to run models using DDP. The DDP examples are explained as part of [Distributed Data Parallel Training Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/distributed_data_parallel.html#neuronx-ddp-tutorial)

To run a basic MLP model on MNIST dataset, please follow the instructions below

## Getting started

```
pip install -U torchvision==0.14.1
torchrun --nproc_per_node=2 mnist_mlp.py
```