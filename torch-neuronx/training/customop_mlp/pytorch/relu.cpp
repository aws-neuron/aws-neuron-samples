#include <stdint.h>
#include <stdlib.h>
#include <torch/torch.h>

torch::Tensor relu_forward(const torch::Tensor& t_in) {
  torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat); 
  auto t_in_acc = t_in.accessor<float, 2>();
  auto t_out_acc = t_out.accessor<float, 2>();
  auto shape = t_in.sizes();
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_in_acc[i][j] : 0.0;
    }
  }
  return t_out;
}

torch::Tensor relu_backward(const torch::Tensor& t_grad, const torch::Tensor& t_in) {
  torch::Tensor t_out = torch::zeros(t_in.sizes(), torch::kFloat); 
  auto t_in_acc = t_in.accessor<float, 2>();
  auto t_grad_acc = t_grad.accessor<float, 2>();
  auto t_out_acc = t_out.accessor<float, 2>();
  auto shape = t_in.sizes();
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      t_out_acc[i][j] = t_in_acc[i][j] > 0.0 ? t_grad_acc[i][j] : 0.0;
    }
  }
  return t_out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("relu_forward", &relu_forward);
  m.def("relu_backward", &relu_backward);
}
