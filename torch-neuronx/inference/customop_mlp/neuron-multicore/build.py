import os
import torch_neuronx
from torch_neuronx.xla_impl import custom_op

custom_op.load(
    name='relu',
    compute_srcs=['relu.cpp'],
    shape_srcs=['shape.cpp'],
    build_directory=os.getcwd(),
    multicore=True,
    verbose=True
)
