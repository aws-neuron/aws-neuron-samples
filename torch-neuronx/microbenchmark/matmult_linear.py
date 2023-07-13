import torch
import torch.nn as nn
import numpy as np
import torch_neuronx
import sys
import logging
import argparse
import ubench_utils

# Setup debug flags
import os

parser = argparse.ArgumentParser(
    description='''Matmult unit test. Executes L consequitive matmults [B,N,N] * [B,N,N]
in a row''')

parser.add_argument('--batch_size',
                    '-b',
                    type=int,
                    metavar="B",
                    help='dim_0 (B) of matrix [B,N,N]',
                    required=True)
parser.add_argument('--matrix_dim',
                    '-n',
                    type=int,
                    metavar="N",
                    help='dim_0, and dim_1 (N) of matrix [B,N,M]',
                    required=True)
parser.add_argument('--loop_count',
                    '-l',
                    type=int,
                    metavar="L",
                    help='Number of times to repeat the matmult',
                    required=True)
parser.add_argument('--num_warmup_iterations',
                    '-w',
                    type=int,
                    metavar="W",
                    help='Number of times to execute the model in the warmup stage',
                    default=8)
parser.add_argument('--num_timed_iterations',
                    '-i',
                    type=int,
                    metavar="I",
                    help='Number of times to execute the model in the timed (i.e., benchmarking) stage',
                    default=1)
parser.add_argument('--num_verification_iterations',
                    type=int,
                    metavar="N",
                    help='Number of times to execute the model in the verification stage',
                    default=2)
parser.add_argument('--neuron_cc_flags',
                    help='optional string containing flags directive for the compiler',
                    default="")
parser.add_argument('--skip_compilation',
                    action='store_true',
                    help='skip compilation, and instead use existing trace')
parser.add_argument('--skip_verification', action='store_true', help='skip verification step')
parser.add_argument('--add_bias',
                    action='store_true',
                    help='add bias to the computation',
)
parser.add_argument('--verbose', '-v', action='store_true', help='increase verbosity level')

args = parser.parse_args()

logging.basicConfig(format='[%(asctime)s %(levelname)s %(name)s:%(lineno)d]  %(message)s')
logger = logging.getLogger()
if (args.verbose):
    logger.setLevel("DEBUG")
else:
    logger.setLevel("INFO")


# Matmult module
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


logger.info('Arguments: ' + ' '.join(f'{k}={v}' for k, v in vars(args).items()))
matrix_cpu = torch.randn([args.batch_size, args.matrix_dim, args.matrix_dim], dtype=torch.float32)

model = Matmult(args.matrix_dim, args.add_bias, args.loop_count)
model.eval()

#Store trace
if args.skip_compilation:
    logger.warning("Skipping compilation. Will use existing trace file")
else:
    logger.info("Starting compilation")
    with ubench_utils.Timer() as compilation_timer:
        trace = torch_neuronx.trace(model,
                                    matrix_cpu,
                                    compiler_workdir='./compiler_dir',
                                    compiler_args=args.neuron_cc_flags)
    torch.jit.save(trace, 'model.pt')
    logger.info("Done with compilation. compilation_time = {:2g}s".format(compilation_timer()))

#Execute on NeuronCore
loaded = torch.jit.load('model.pt')

#Warmup
logger.info("Starting warmup")
with ubench_utils.Timer() as warmup_timer:
    for i in range(args.num_warmup_iterations):
        out = loaded(matrix_cpu)
logger.info("Done with warmup. warmup_time = {:2g}s, num_warmup_iterations = {}".format(
    warmup_timer(), args.num_warmup_iterations))
logger.info(
    "Result = {} (printing here to force computation; there is no meaning to this number)".format(
        out))




#Timed Run:
logger.info("Starting timed run")
with ubench_utils.Timer() as benchmark_timer:
    for i in range(args.num_timed_iterations):
        out = loaded(matrix_cpu)
logger.info(
    "Done with timed run. overall_runtime = {:2g}s, runtime_per_iteration = {:2g}s, num_timed_iterations = {}"
    .format(benchmark_timer(),
            benchmark_timer() / args.num_timed_iterations, args.num_timed_iterations))


top_per_run = args.batch_size*(args.matrix_dim**3)*args.num_timed_iterations*args.loop_count*2
tops = (top_per_run/benchmark_timer())/1e12
logger.info("PE TOPS = {:2g}".format(tops))




#Verify results:
if args.skip_verification:
    logger.warning("Skipping verification step")
if not args.skip_verification:
    logger.info("Starting verification runs")
    verfication_res = []
    with ubench_utils.Timer() as benchmark_timer:
        for i in range(args.num_verification_iterations):
            verfication_res.append(loaded(matrix_cpu))

    # Compare runs on device against themselves
    for i in range(1, args.num_verification_iterations):
        logger.debug("result[{}] = {}, result[{}] = {}".format(i, verfication_res[i], i - 1,
                                                               verfication_res[i - 1]))
        np.testing.assert_allclose(verfication_res[i], verfication_res[i - 1])

    logger.info("Done with verification")

logger.debug("matrix_cpu.shape={}".format(matrix_cpu.shape))
logger.debug("out shape={}".format(out.shape))

logger.info("Done!")
