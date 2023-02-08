# Torch Neuron CustomOp MLP

This folder contains examples Torch custom operators for a multi-layer perceptron (MLP) model.

- The `pytorch` folder contains a basic PyTorch (non-neuron) CPU-based MLP model with a custom Relu operator and training script.
- The `neuron` folder contains a the same model but converted to Neuron with an XLA-based training script for trn1-based instances.