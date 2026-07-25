# Change Log

## September, 15th 2023
* Added notebook script to fine-tune ``deepmind/language-perceiver`` model using ``torch-neuronx``. 
* Added notebook script to fine-tune ``clip-large`` model using ``torch-neuronx``.
* Added ``SD XL Base+Refiner`` inference sample script using ``torch-neuronx``.
* Upgraded default ``diffusers`` library from 0.14.0 to latest 0.20.2 in ``Stable Diffusion 1.5`` and ``Stable Diffusion 2.1`` inference scripts. 
* Removed the deprecated  ``--model-type=transformer-inference`` flag from ``Llama-2-13B`` model inference sample using ``transformers-neuronx``



## August, 28th 2023
* Added sample script for LLaMA V2 13B model inference using transformers-neuronx
* Added samples for training GPT-NEOX 20B and 6.9B models using neuronx-distributed 
* Added sample scripts for CLIP and Stable Diffusion XL inference using torch-neuronx
* Added sample scripts for vision and language Perceiver models inference using torch-neuronx
* Added camembert training/finetuning example for Trn1 under hf_text_classification in torch-neuronx
* Updated Fine-tuning Hugging Face BERT Japanese model sample in torch-neuronx
* Updated OPT and GPT-J transformers-neuronx inference samples to install transformers-neuronx from whl instead of using github repo
* Upgraded numpy package to 1.21.6 in GPT-2 and several training samples under hf_text_classification in torch-neuronx
* Removed pinning of torch-neuron and tensorflow-neuron libraries and other minor changes in several of torch-neuron and tensorflow-neuron Inf1 inference samples.


## February, 23rd 2023
* Added OPT-13B, OPT-30B, OPT-66B inference examples under transformers-neuronx
* Added distilbert-base-uncased training/finetuning example for Trn1 under torch-neuronx

## November, 7th 2022

* Added Fine-tuning Hugging Face BERT Japanese model sample

## November,4th 2022
* Added HuggingFace Vision Transformer (ViT)training examples for Trn1 under torch-neuronx.

## October,27th 2022
* Added HuggingFace GPT2 training examples for Trn1 under torch-neuronx.
* Added 7 Pytorch training examples for Trn1 under torch-neuronx.

## October,10th 2022

* Added 20 Pytorch inference examples for Inf1 under torch-neuron.
* Added 1 TensorFlow inference example for Inf1 under tensorflow-neuron.
* Added 2 Pytorch inference examples for Inf1 under torch-neuronx.

# Known Issues

* With 2.7 release of Neuron SDK, there is a known low accuracy with Hugging Face Roberta Large, finetuning on trn1
* for BF16 version of the model.

