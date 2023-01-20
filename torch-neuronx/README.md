# PyTorch Neuron (torch-neuronx) Samples for AWS Trn1

This directory contains sample PyTorch Neuron inference and training scripts that can be run on [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (Trn1 instances).

For additional information on these training scripts, please refer to the tutorials found in the <mark>official Trainium documentation</mark>.

## Training

The following samples are available for training:

| Name                                                        | Description                                                                                                                             | Training Parallelism |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------| --- |
| [dp_bert_hf_pretrain](training/dp_bert_hf_pretrain)         | Phase1 and phase2 pretraining of Hugging Face BERT-large model                                                                          | DataParallel |
| [mnist_mlp](training/mnist_mlp)                             | Examples of training a multilayer perceptron on the MNIST dataset                                                                       | DataParallel |
| [hf_text_classification](training/hf_text_classification)   | Fine-tuning various Hugging Face models for a text classification task                                                                  | DataParallel |
| [hf_image_classification](training/hf_image_classification) | Fine-tuning Hugging Face models (ex. ViT) for a image classification task                                                               | DataParallel |
| [hf_language_modeling](training/hf_language_modeling)       | Training Hugging Face models (ex. GPT2) for causal language modeling (CLM)                                                              | DataParallel |
| [hf_bert_jp](training/hf_bert_jp_text_classification)       | Fine-tuning Hugging Face BERT Japanese model                                                                                            | DataParallel |
| [hf_sentiment_analysis](training/hf_sentiment_analysis)     | Examples of training Hugging Face bert-base-cased model for a text classification task with Trn1 Single Neuron and Distributed Training | DataParallel |

## Inference

The following samples are available for inference:

| Model Name                                                        | Model Task                                                                                                                             | Original Model Source |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------| --- |
| [BERT Base Uncased](inference/hf_pretrained_bert_inference_on_trn1.ipynb)         | Masked language modeling and next sentence prediction                                                                          | [bert-base-uncased](https://huggingface.co/bert-base-uncased) |
| [DistilBERT Base Uncased](inference/hf_pretrained_distilbert_Inference_on_trn1.ipynb)         | Masked language modeling and next sentence prediction                                                                          | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) |
| [RoBERTa Large](inference/hf_pretrained_roberta_inference_on_frn1.ipynb)         | Masked language modeling, sequence classification, and question and answering                                                                          | [roberta-large](https://huggingface.co/roberta-large)  |
| [GPT2](inference/hf_pretrained_gpt2_feature_extraction_on_trn1.ipynb)         | Text feature extraction                                                                          | [gpt2](https://huggingface.co/gpt2) |
| [Vision Transformer (ViT)](inference/hf_pretrained_vit_inference_on_trn1.ipynb)         | Image classification                                                                          | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) |
| [ResNet50](inference/tv_pretrained_resnet50_inference_on_trn1.ipynb)         | Image classification                                                                       | [resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) |
