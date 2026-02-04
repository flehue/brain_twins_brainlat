#!/bin/sh
#SBATCH -t3-12
#SBATCH --mail-type=FAIL,TIME_LIMIT_80
#SBATCH --output=slurm-%A.out
#SBATCH -c 1
#SBATCH --mem-per-cpu=4G
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 $@