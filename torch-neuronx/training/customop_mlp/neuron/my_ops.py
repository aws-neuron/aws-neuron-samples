import torch
import torch_neuronx
from torch_neuronx.xla_impl import custom_op

custom_op.load_library('librelu.so')

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ops.my_ops.relu_forward(input)

    @staticmethod
    def backward(ctx, grad):
        input, = ctx.saved_tensors
        return torch.ops.my_ops.relu_backward(grad, input), None


