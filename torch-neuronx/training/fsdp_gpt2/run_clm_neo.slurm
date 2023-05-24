#!/bin/bash
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
	 
srun ./run_clm_neo.sh MIXED wikitext-103-raw-v1
