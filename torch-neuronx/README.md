# PyTorch Neuron (torch-neuronx) Samples for AWS Trn1

This directory contains sample PyTorch Neuron training scripts that can be run on [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (trn1 instances).

For additional information on these training scripts, please refer to the tutorials found in the <mark>official Trainium documentation</mark>.

The following samples are available:

| Name                                                      | Description                                                                                                                            | Training Parallelism |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| [dp_bert_hf_pretrain](training/dp_bert_hf_pretrain)       | Phase1 and phase2 pretraining of Hugging Face BERT-large model                                                                         | DataParallel         |
| [mnist_mlp](training/mnist_mlp)                           | Examples of training a multilayer perceptron on the MNIST dataset                                                                      | DataParallel         |
| [hf_sentiment_analysis](training/hf_sentiment_analysis)   | Fine-tuning various Hugging Face models for a text classification task                                                                 | DataParallel         |
| [hf_text_classification](training/hf_text_classification) | Exaples of training Hugging Face bert-base-cased model for a text classification task with Trn1 Single Neuron and Distributed Training | ModelParallel        |
