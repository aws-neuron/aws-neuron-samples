#!/bin/bash
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodelist=compute1-st-kaena-training-0-[1-4]

srun neuron_parallel_compile ./run_clm.sh MIXED wikitext-103-raw-v1
