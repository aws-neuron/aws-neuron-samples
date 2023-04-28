import os
import time
import torch
from model import MLP

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
import torch_xla.core.xla_model as xm

# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 32

# Load MNIST inference dataset
inf_dataset = mnist.MNIST(root='./MNIST_DATA_inf',
                            train=False, download=True, transform=ToTensor())

def main():
    # Prepare data loader
    inf_loader = DataLoader(inf_dataset, batch_size=BATCH_SIZE)

    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = 'xla'

    # Init with random weight and move model to device
    model = MLP()
    torch.nn.init.xavier_normal_(model.fc1.weight)
    torch.nn.init.xavier_normal_(model.fc2.weight)
    torch.nn.init.xavier_normal_(model.fc3.weight)
    model = model.to(device)

    # Run the training loop
    print('---------- Inference ---------------')
    model.eval()
    for _ in range(EPOCHS):
        start = time.time()
        for idx, (inf_x, _) in enumerate(inf_loader):
            inf_x = inf_x.view(inf_x.size(0), -1)
            inf_x = inf_x.to(device)
            output = model(inf_x)
            xm.mark_step() # XLA: collect ops and run them in XLA runtime
            if idx < WARMUP_STEPS: # skip warmup iterations
                start = time.time()
    # Compute statistics for the last epoch
    interval = idx - WARMUP_STEPS # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Inf throughput (iter/sec): {}".format(throughput))

    print('----------End Inference ---------------')

if __name__ == '__main__':
    main()

