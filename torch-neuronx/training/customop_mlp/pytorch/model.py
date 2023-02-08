import torch
import torch.nn as nn
from torch.nn import functional as F
import my_ops

# Declare 3-layer MLP for MNIST dataset
class MLP(nn.Module):
  def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, layers[0])
    self.fc2 = nn.Linear(layers[0], layers[1])
    self.fc3 = nn.Linear(layers[1], output_size)

  def forward(self, x):
    f1 = self.fc1(x)
    r1 = my_ops.Relu.apply(f1)
    f2 = self.fc2(r1)
    r2 = my_ops.Relu.apply(f2)
    f3 = self.fc3(r2)
    return torch.log_softmax(f3, dim=1)
