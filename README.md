# AWS Neuron Samples

This repository contains samples for [AWS Neuron](https://aws.amazon.com/machine-learning/neuron/), the software development kit (SDK) that enables machine learning (ML) inference and training workloads on the AWS ML accelerator chips [Inferentia](https://aws.amazon.com/machine-learning/inferentia/) and [Trainium](https://aws.amazon.com/machine-learning/trainium/).

The samples in this repository provide an indication of the types of deep learning models that can be used with Trainium and Inferentia, but do not represent an exhaustive list of supported models. If you have additional model samples that you would like to contribute to this repository, please submit a pull request following the repository's contribution [guidelines](CONTRIBUTING.md).

Samples are organized by use case (training, inference) and deep learning framework (PyTorch, TensorFlow) below:

## Training

| Framework | Description | Instance Type |
| --- | --- | --- |
| [PyTorch Neuron (torch-neuronx)](torch-neuronx/README.md#training) | Sample training scripts for training various PyTorch models on AWS Trainium | Trn1, Trn1n & Inf2 |

| Usage | Description | Instance Type |
| --- | --- | --- |
| [Megatron-LM for Neuron](https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm) | A library that enables large-scale distributed training of language models such as GPT and is adapted from Megatron-LM. | Trn1, Trn1n |
| [AWS Neuron samples for ParallelCluster](ihttps://github.com/aws-neuron/aws-neuron-parallelcluster-samples) | How to use AWS ParallelCluster to build HPC compute cluster that uses trn1 compute nodes to run your distributed ML training job.  | Trn1, Trn1n |
| [AWS Neuron samples for EKS](https://github.com/aws-neuron/aws-neuron-eks-samples) | iThe samples in this repository demonstrate the types of patterns that can be used to deliver inference and distributed training on EKS using Inferentia and Trainium. | Trn1, Trn1n |
| [AWS Neuron samples for SageMaker](https://github.com/aws-neuron/aws-neuron-sagemaker-samples) | SageMaker Samples using ml.inf2 and ml.trn1 instances for machine learning (ML) training workloads on the AWS ML accelerator chips Trainium. | Trn1, Trn1n |


## Inference

| Framework | Description | Instance Type |
| --- | --- | --- |
| [PyTorch Neuron (torch-neuron)](torch-neuron) | Sample Jupyter notebooks demonstrating model compilation and inference for various PyTorch models on AWS Inferentia | Inf1 |
| [PyTorch Neuron (torch-neuronx)](torch-neuronx/README.md#inference) | Sample Jupyter notebooks demonstrating model compilation and inference for various PyTorch models on AWS Inferentia2 and Trainium | Inf2 & Trn1 |
| [PyTorch Neuron (transformers-neuronx)](torch-neuronx/transformers-neuronx) | Sample Jupyter Notebooks demonstrating tensor parallel inference for various PyTorch large language models (LLMs) on AWS Inferentia and Trainium | Inf2 & Trn1 |
| [TensorFlow Neuron (tensorflow-neuron)](tensorflow-neuron) | Sample Jupyter notebooks demonstrating model compilation and inference for various TensorFlow models on AWS Inferentia | Inf1 |

| Usage | Description | Instance Type |
| --- | --- | --- |
| [AWS Neuron samples for SageMaker](https://github.com/aws-neuron/aws-neuron-sagemaker-samples) | SageMaker Samples using ml.inf2 and ml.trn1 instances for machine learning (ML) inference workloads on the AWS ML accelerator chips Inferentia and Trainium.  | Inf2 & Trn1 |


## Getting Help

If you encounter issues with any of the samples in this repository, please open an issue via the GitHub Issues feature.

## Contributing

Please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for details on contributing additional samples to this repository.


## Release Notes

Please refer to the [Change Log](releasenotes.md).

## Known Issues

| Model | Framework | Training/Inference | Instance Type | Status |
| --- | --- | --- | --- | --- |
| Fairseq | PyTorch | Inference | Inf1 | RuntimeError: No operations were successfully partitioned and compiled to neuron for this model - aborting trace! |
| Yolof | PyTorch | Inference | Inf1 | RuntimeError: No operations were successfully partitioned and compiled to neuron for this model - aborting trace! |
