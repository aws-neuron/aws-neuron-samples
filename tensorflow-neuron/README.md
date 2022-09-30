# TensorFlow Neuron Samples for AWS Inferentia

This directory contains Jupyter notebooks that demonstrate model compilation and inference using TensorFlow Neuron for a variety of popular deep learning models. These samples can be run on [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) (inf1 instances) using [Amazon SageMaker](https://aws.amazon.com/sagemaker) or [Amazon EC2](https://aws.amazon.com/ec2/).

For each sample you will also find additional information such as the model type, configuration used to compile the model, framework version, and a link to the original model implementation.

The following samples are available:

|Model Name	|Model Type	|Input Shape	|NeuronSDK Version	|Framework / Version	|Original Implementation	|
|---	|---	|---	|---	|---	|---	|
|[U-Net](inference/unet) |CV - Semantic Segmentation    |1,3,224,224    |2.5.2.2.1.14.0    |Tensorflow 2.5.2    |[link](https://github.com/jakeret/unet)|


### Configuring the environment

In order to run the samples, you first need to [set up a TensorFlow Neuron development environment](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html).

