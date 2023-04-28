#include <stdint.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "torchneuron/register.h"

torch::Tensor relu_fwd_shape(torch::Tensor t_in) {
    torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat);
    return t_out;
}

torch::Tensor relu_bwd_shape(torch::Tensor t_grad, torch::Tensor t_in) {
    torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat);
    return t_out;
}

NEURON_LIBRARY(my_ops, m) {
  m.def("relu_forward", &relu_fwd_shape, "relu_forward");
  m.def("relu_backward", &relu_bwd_shape, "relu_backward");
}
