import os
import torch
from torch.utils import cpp_extension

cpp_extension.load(
    name='librelu',
    sources=['relu.cpp'],
    is_python_module=False,
    build_directory=os.getcwd()
)
