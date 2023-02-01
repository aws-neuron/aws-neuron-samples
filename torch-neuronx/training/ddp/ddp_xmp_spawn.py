import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend


def setup(init_file: str):
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        "xla",
        init_method=f"file://{init_file}" if init_file is not None else None,
        rank=rank,
        world_size=world_size)
    return rank, world_size


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train_fn(rank):
    rank, _ = setup(None)
    device = xm.xla_device()
    
    # Create the model and move to device
    model = Model().to(device)
    ddp_model = DDP(model, gradient_as_bucket_view=True)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    num_iterations = 100
    for step in range(num_iterations):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(device))
        labels = torch.randn(20, 5).to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        xm.mark_step()
        if rank == 0:
            print(f"Loss after step {step}: {loss.cpu()}")

        
def run():
    xmp.spawn(train_fn, args=())

if __name__ == '__main__':
    run()