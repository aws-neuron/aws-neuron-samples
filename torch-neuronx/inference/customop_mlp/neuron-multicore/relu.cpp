#include <stdint.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <neuron/neuron-utils.hpp>

torch::Tensor relu_forward(const torch::Tensor& t_in) {
  size_t num_elem = t_in.numel();
  torch::Tensor t_out = get_dst_tensor();

  uint32_t cpu_id = get_cpu_id();
  uint32_t cpu_count = get_cpu_count();
  uint32_t partition = num_elem / cpu_count;
  if (cpu_id == cpu_count - 1) {
    partition = num_elem - partition * (cpu_count - 1);
  }

  static constexpr size_t buffer_size = 1024;
  float *tcm_buffer = (float*)torch::neuron::tcm_malloc(sizeof(float) * buffer_size);

  if (tcm_buffer != nullptr) {
    auto t_in_tcm_acc = t_in.tcm_accessor();
    auto t_out_tcm_acc = t_out.tcm_accessor();

    for (size_t i = 0; i < partition; i += buffer_size) {
      size_t remaining_elem = partition - i;
      size_t copy_size = (remaining_elem > buffer_size) ? buffer_size : remaining_elem;

      t_in_tcm_acc.tensor_to_tcm<float>(tcm_buffer, partition *cpu_id + i, copy_size);
      for (size_t j = 0; j < copy_size; j++) {
          tcm_buffer[j] = tcm_buffer[j] > 0.0 ? tcm_buffer[j] : 0.0;
      }
      t_out_tcm_acc.tcm_to_tensor<float>(tcm_buffer, partition *cpu_id + i, copy_size);
    }
  }
  torch::neuron::tcm_free(tcm_buffer);
  return t_out;
}
