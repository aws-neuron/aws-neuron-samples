#!/bin/bash
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
	 
srun ./run_clm_slurm.sh MIXED wikitext-103-raw-v1
