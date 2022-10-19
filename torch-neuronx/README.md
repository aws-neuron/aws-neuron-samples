# PyTorch Neuron (torch-neuronx) Samples for AWS Trn1

This directory contains sample PyTorch Neuron training scripts that can be run on [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (trn1 instances).

For additional information on these training scripts, please refer to the tutorials found in the <mark>official Trainium documentation</mark>.

The following samples are available:

| Name | Description | Training Parallelism |
| --- | --- | --- |
| [dp_bert_hf_pretrain](training/dp_bert_hf_pretrain) | Phase1 and phase2 pretraining of Hugging Face BERT-large model | DataParallel |
| [mnist_mlp](training/mnist_mlp) | Examples of training a multilayer perceptron on the MNIST dataset | DataParallel |
| [hf_text_classification_bert_base_cased](training/hf_text_classification/BertBaseCased.ipynb) | Example of fine-tuning a Hugging Face bert-base-cased model for a text classification task | DataParallel |
| [hf_text_classification_bert_base_uncased](training/hf_text_classification/BertBaseUncased.ipynb) | Example of fine-tuning a Hugging Face bert-base-uncased model for a text classification task | DataParallel |
| [hf_text_classification_bert_large_cased](training/hf_text_classification/BertLargeCased.ipynb) | Example of fine-tuning a Hugging Face bert-large-cased model for a text classification task | DataParallel |
| [hf_text_classification_bert_large_uncased](training/hf_text_classification/BertLargeUncased.ipynb) | Example of fine-tuning a Hugging Face bert-large-uncased model for a text classification task | DataParallel |
| [hf_text_classification_roberta_base](training/hf_text_classification/RobertaBase.ipynb) | Example of fine-tuning a Hugging Face roberta-base model for a text classification task | DataParallel |
| [hf_text_classification_roberta_large](training/hf_text_classification/RobertaLarge.ipynb) | Example of fine-tuning a Hugging Face roberta-large model for a text classification task | DataParallel |
| [hf_text_classification_xlm_roberta_base](training/hf_text_classification/XlmRobertaBase.ipynb) | Example of fine-tuning a Hugging Face xlm-roberta-base model for a text classification task | DataParallel |