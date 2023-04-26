# Torch Neuron CustomOp MLP

This folder contains inference examples Torch custom operators for a multi-layer perceptron (MLP) model.

- The `neuron` folder contains a MLP model with relu implemented as a CustomOp using element-wise accessor.
- The `neuron-tcm` folder contains the same model but relu is implemented using tcm accessor.
- The `neuron-multicore` folder contains the same model but relu is implemeted using tcm accessor and multicore capability.