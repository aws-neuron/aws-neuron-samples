import torch

torch.ops.load_library('librelu.so')

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.ops.my_ops.relu_forward(input)

    @staticmethod
    def backward(ctx, grad):
        input, = ctx.saved_tensors
        return torch.ops.my_ops.relu_backward(grad, input), None


