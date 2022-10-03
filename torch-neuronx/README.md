# PyTorch Neuron (torch-neuronx) Samples for AWS Trn1

This directory contains sample PyTorch Neuron training scripts that can be run on [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (trn1 instances).

For additional information on these training scripts, please refer to the tutorials found in the <mark>official Trainium documentation</mark>.

The following samples are available:

| Name | Description | Training Parallelism |
| --- | --- | --- |
| [dp_bert_hf_pretrain](training/dp_bert_hf_pretrain) | Phase1 and phase2 pretraining of Hugging Face BERT-large model | DataParallel |
| [mnist_mlp](training/mnist_mlp) | Examples of training a multilayer perceptron on the MNIST dataset | DataParallel |
