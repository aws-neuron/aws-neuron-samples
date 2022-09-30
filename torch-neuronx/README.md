# PyTorch Neuron Samples for AWS Trainium

This directory contains sample PyTorch Neuron training scripts that can be used on AWS Trainium (trn1 instances).

The following samples are available:

| Name | Description | Training Parallelism |
| --- | --- | --- |
| [dp_bert_hf_pretrain](training/dp_bert_hf_pretrain) | Phase1 and phase2 pretraining of Hugging Face BERT-large model. | DataParallel |
| [mnist_mlp](training/mnist_mlp) | Examples of training a multilayer perceptron on the MNIST dataset. | DataParallel |
