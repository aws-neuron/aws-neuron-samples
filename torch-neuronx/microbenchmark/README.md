# Neuron Microbenchmark Guide   

## Overview

This guide reviews the best practices for benchmarking performance of Neuron devices. It shows how to separate compilation and execution time, how to isolate the device time from the end-to-end execution time, how to warm-up the device, and covers few pitfalls one should be aware of.  This guide provides an example code, in PyTorch, that can be used as a template for measuring performance.  

## Example 

As a motivating example, assume we would like to measure the max throughput of the device when executing matrix multiplication:

```
nn.Linear(in_features=n, out_features=n, bias=is_add_bias)
```

Note that `nn.Linear` can add bias; we will touch on that part later.   


## Initial Version 

Let’s write a simple Module that will exercise the Linear layer in a loop (see below). We want to repeat the computation to amortize overheads.  

```
import torch
import torch.nn as nn

@torch.no_grad()
class Matmult(nn.Module):

    def __init__(self, n, is_add_bias, loop_count):
        super().__init__()
        self.loop_count = loop_count
        self.matmult = nn.Linear(in_features=n, out_features=n, bias=is_add_bias)

    def forward(self, x):
        out = self.matmult(x)
        for i in range(1, self.loop_count):
            out = self.matmult(out)
        return out.mean()

```

Note that we feed the result of the previous matmult to the current one. This is done to make sure we use the result from each matrix multiplication. If, for example, we would have tried to simply repeat the same computation inside the loop, the compiler would have optimized all but the last iteration out

```
    def forward(self, x):
        input = x
        for i in range(0, self.loop_count):
            out = self.matmult(input) 
```



## Using PyTorch-Neuron trace

There are two methods to instantiate execution on neuron devices: (1) using [Neuron XLA device API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide.html), and (2) using [PyTorch-Neuron trace API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/api-compilation-python-api.html). For benchmarking, we prefer using the PyTorch-Neuron trace, because it introduces minimal runtime and application overheads (see explanation of the [Lazy mode](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide.html#understand-the-lazy-mode-in-pytorch-neuron) operation of Neuron XLA).

PyTorch-Neuron trace also makes it easy to separate compilation and execution:

```
...
import torch_neuronx

matrix_cpu = torch.randn([args.batch_size, args.matrix_dim, args.matrix_dim], dtype=torch.float32)
model = Matmult(args.matrix_dim, args.add_bias, args.loop_count)

#Compile model
trace = torch_neuronx.trace(model,
                            (matrix_cpu),
                            compiler_workdir='./compiler_dir',
                            compiler_args=args.neuron_cc_flags)

# Save model to disk 
torch.jit.save(trace, 'model.pt')

# Load model on NeuronCore
neuron_model = torch.jit.load('model.pt')

# Warmup
out = neuron_model(matrix_cpu)

# Timed run
for i in range(REPEATED_RUNS):
    out = neuron_loaded(matrix_cpu)
```

## Counting time

Make sure to use a sufficiently-granular counter. We recommend using [`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)`, `which uses the clock with the highest available resolution. The Neuron microbenchmark samples, contain a simple [utility](ubench_utils.py) that is adequate for perf timing. Using the timer class, we can decorate the previous code to measure runtime of each section. 


```
...
import ubench_utils 

#Compile model
with ubench_utils.Timer() as compilation_time:
    trace = torch_neuronx.trace(model, 
                                matrix_cpu, 
                                compiler_args=args.neuron_cc_flags)
                                compiler_args=compiler_args)

# Save model to disk 
torch.jit.save(trace, 'model.pt')

# Load model on NeuronCore
neuron_model = torch.jit.load('model.pt')

# Warmup
with ubench_utils.Timer() as warmup_model_time:
    out = neuron_model(matrix_cpu)

# Timed run
with ubench_utils.Timer() as benchmark_time:
    for i in range(args.num_timed_iterations):
        out = neuron_loaded(matrix_cpu)
        
        
print("""compilation took {}s, warmup took {}s, benchmark took {}s"""
     .format(compilation_time(), 
             warmup_model_time(), 
             benchmark_time()))             
```

 
## Full example

A complete, parametrizable example of matrix multiplication benchmarks is in [matmult_linear.py](matmult_linear.py). It allows setting the batch size, matrix size, loop and iteration count, as well as additional parameters (listed using `python matmult_linear.py -h`). Example usage:

```
python matmult_linear.py --batch_size 1 --matrix_dim 1024 --loop_count 1000 --num_warmup_iterations 2 --num_timed_iterations 1000 --add_bias
```

We recommend using `--add_bias` for numerical stability (avoiding NaNs in computation). Numerical issues are reported back to the user, which can slow down total runtime. For best performance use large matrix sizes (for high utilization), and large loop/iteration counts (to minimize overheads).



   





  

