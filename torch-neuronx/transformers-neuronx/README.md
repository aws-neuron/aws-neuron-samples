# PyTorch Neuron (transformers-neuronx) Samples for AWS Inf2 & Trn1

This directory contains sample Jupyter Notebooks demonstrating tensor parallel inference for various PyTorch large language models (LLMs) on [AWS Inferentia](https://aws.amazon.com/ec2/instance-types/inf2/) (Inf2) instances) and [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (Trn1) instances.

For additional information on these training scripts, please refer to the tutorials found in the <mark>official Inferentia and Trainium documentation</mark>.

## Inference

The following samples are available for LLM tensor parallel inference:

| Name                                                        | Instance type |
|-------------------------------------------------------------| --------------- |
| [facebook/opt-13b](inference/facebook-opt-13b-sampling.ipynb) | Inf2 & Trn1 |
| [facebook/opt-30b](inference/facebook-opt-30b-sampling.ipynb) | Inf2 & Trn1 |
| [facebook/opt-66b](inference/facebook-opt-66b-sampling.ipynb) | Inf2 |
| [meta-llama/Llama-2-13b](inference/meta-llama-2-13b-sampling.ipynb) | Inf2 & Trn1 |
| [codellama/CodeLlama-7b-hf](inference/CodeLlama-7B.ipynb) | Inf2 & Trn1|
