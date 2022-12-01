#!/bin/bash
#SBATCH --job-name=main.py.job
#SBATCH --output=./slurmRun/run.out
#SBATCH --error=./slurmRun/run.err
#SBATCH --gres=gpu:1
#SBATCH --partition=titan-giant    # Partition to submit to 
#SBATCH --mem=12000
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@cs.utah.edu
python ./main.py 
exit
