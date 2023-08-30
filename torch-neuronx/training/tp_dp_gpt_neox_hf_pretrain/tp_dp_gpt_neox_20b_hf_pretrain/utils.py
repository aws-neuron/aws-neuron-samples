import torch
from neuronx_distributed.parallel_layers import mappings

class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""
    @staticmethod
    def symbolic(graph, input_):
        return mappings._split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return mappings._split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return mappings._gather_along_first_dim(grad_output)


# Note: This function is going to be upstreamed to Neuronx-Distributed in the upcoming release.
def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)