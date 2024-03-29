# PyTorch Neuron (torch-neuronx) Samples for AWS Trn1/Trn1n & Inf2 Instances

This directory contains sample PyTorch Neuron inference and training scripts that can be run on [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (Trn1/Trn1n instances) and [AWS Inferentia]( https://aws.amazon.com/machine-learning/inferentia/) (Inf2 instances).

For additional information on these training scripts, please refer to the tutorials found in the <mark>official Trainium documentation</mark>.

## Training

The following samples are available for training:

| Name                                                        | Description                                                                                                                             | Training Parallelism |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------| --- |
| [dp_bert_hf_pretrain](training/dp_bert_hf_pretrain)         | Phase1 and phase2 pretraining of Hugging Face BERT-large model                                                                          | DataParallel |
| [mnist_mlp](training/mnist_mlp)                             | Examples of training a multilayer perceptron on the MNIST dataset                                                                       | DataParallel |
| [mnist_mlp](training/ddp)                                   | Examples of training a multilayer perceptron on the MNIST dataset using DDP                                                             | DataParallel |
| [hf_text_classification](training/hf_text_classification)   | Fine-tuning various Hugging Face models for a text classification task                                                                  | DataParallel |
| [hf_image_classification](training/hf_image_classification) | Fine-tuning Hugging Face models (ex. ViT) for a image classification task                                                               | DataParallel |
| [hf_contrastive_image_text](training/hf_contrastive_image_text) | Fine-tuning Multi-modal Image and Text Hugging Face models (ex. CLIP)                                                                 | DataParallel |
| [hf_language_modeling](training/hf_language_modeling)       | Training Hugging Face models (ex. GPT2) for causal language modeling (CLM)                                                              | DataParallel |
| [hf_bert_jp](training/hf_bert_jp)                           | Fine-tuning & Deployment Hugging Face BERT Japanese model                                                                               | DataParallel |
| [hf_sentiment_analysis](training/hf_sentiment_analysis)     | Examples of training Hugging Face bert-base-cased model for a text classification task with Trn1 Single Neuron and Distributed Training | DataParallel |
| [customop_mlp](training/customop_mlp)     | Examples of training a multilayer perceptron model with a custom Relu operator on a single Trn1 | DataParallel |
| [tp_dp_gpt_neox_20b_hf_pretrain](https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain) | Training GPT-NEOX 20B model using neuronx-distributed | Tensor Parallel & DataParallel |
| [tp_dp_gpt_neox_6.9b_hf_pretrain](https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain) | Training GPT-NEOX 6.9B model using neuronx-distributed | Tensor Parallel & DataParallel |
| [tp_zero1_llama2_7b_hf_pretrain](https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain) | Training Llama-2 7B model using neuronx-distributed | Tensor Parallel |
| [tp_pp_llama2_70b_hf_pretrain](https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/training/llama2/tp_pp_llama2_hf_pretrain) | Training Llama-2 70B model using neuronx-distributed | Tensor Parallel & Pipeline Parallel |

## Inference

The following samples are available for inference:

| Model Name                                                        | Model Task                                                                                                                             | Original Model Source |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------| --- |
| [BERT Base Uncased](inference/hf_pretrained_bert_inference_on_trn1.ipynb)         | Masked language modeling and next sentence prediction                                                                          | [bert-base-uncased](https://huggingface.co/bert-base-uncased) |
| [DistilBERT Base Uncased](inference/hf_pretrained_distilbert_Inference_on_trn1.ipynb)         | Masked language modeling and next sentence prediction                                                                          | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) |
| [RoBERTa Large](inference/hf_pretrained_roberta_inference_on_frn1.ipynb)         | Masked language modeling, sequence classification, and question and answering                                                                          | [roberta-large](https://huggingface.co/roberta-large)  |
| [Vision Transformer (ViT)](inference/hf_pretrained_vit_inference_on_inf2.ipynb)         | Image classification                                                                          | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) |
 [GPT2](inference/hf_pretrained_gpt2_feature_extraction_on_trn1.ipynb)         | Text feature extraction                                                                          | [gpt2](https://huggingface.co/gpt2) |
| [ResNet50](inference/tv_pretrained_resnet50_inference_on_trn1.ipynb)         | Image classification                                                                       | [resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) |
| [HuggingFace Stable Diffusion 1.5 (512x512)](inference/hf_pretrained_sd15_512_inference.ipynb)         | Text to image generation                                                                       | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| [HuggingFace Stable Diffusion 2.1 (512x512)](inference/hf_pretrained_sd2_512_inference.ipynb)         | Text to image generation                                                                       | [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) |
| [HuggingFace Stable Diffusion 2.1 (768x768)](inference/hf_pretrained_sd2_768_inference.ipynb)         | Text to image generation                                                                      | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| [HuggingFace Stable Diffusion XL Base 1.0 (1024x1024)](inference/hf_pretrained_sdxl_base_1024_inference.ipynb)         | Text to image generation                                                                      | [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| [HuggingFace Stable Diffusion XL Base & Refiner 1.0 (1024x1024)](inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb)         | Text to image generation                                                                      | [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| [UNet](inference/pretrained_unet_inference_on_trn1.ipynb)         | Image Segmentation                                                                         | [unet](https://github.com/milesial/Pytorch-UNet) |
| [VGG](inference/tv_pretrained_vgg_inference_on_trn1.ipynb)         | Image Classification                | [vgg](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html) |
| [Multimodal Perceiver](inference/hf_pretrained_perceiver_multimodal_inference.ipynb)         | Video Classification and Autoencoding               | [multimodal-perceiver](https://huggingface.co/deepmind/multimodal-perceiver) |
| [Language Perceiver](inference/hf_pretrained_perceiver_language_inference.ipynb)         | Text Classification              | [language-perceiver](https://huggingface.co/deepmind/language-perceiver) |
| [Vision Perceiver](inference/hf_pretrained_perceiver_vision_inference.ipynb)         | Image Classification              | [vision-perceiver-conv](https://huggingface.co/deepmind/vision-perceiver-conv) |
| [CLIP Base](inference/hf_pretrained_clip_base_inference_on_inf2.ipynb)         | Image Classification                | [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| [CLIP Large](inference/hf_pretrained_clip_large_inference_on_inf2.ipynb)         | Image Classification               | [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |

The following samples are available for LLM tensor parallel inference:

| Name                                                        | Instance type |
|-------------------------------------------------------------| --------------- |
| [facebook/opt-13b](transformers-neuronx/inference/facebook-opt-13b-sampling.ipynb) | Inf2 & Trn1 |
| [facebook/opt-30b](transformers-neuronx/inference/facebook-opt-30b-sampling.ipynb) | Inf2 & Trn1 |
| [facebook/opt-66b](transformers-neuronx/inference/facebook-opt-66b-sampling.ipynb) | Inf2 |
| [meta-llama/Llama-2-13b](transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb) | Inf2 & Trn1 |

## Microbenchmarking

The following samples are available for microbenchmarking:

| Name                                                        | Description                                                                                                                             
|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------- |
| [tutorial](microbenchmark/microbenchmark.ipynb)         | Microbenchmarking tutorial
| [matmult](microbenchmark/matmult_linear.py)         | Matrix multiplication microbenchmark 
