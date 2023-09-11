#!/bin/bash
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=low-priority
#SBATCH --output=../sbatches/logs/debugging/earlystop_resnet50.out
#SBATCH --error=../sbatches/logs/debugging/earlystop_resnet50.err
#SBATCH --array=29

python ../experiment_grid_search.py --exparray $SLURM_ARRAY_TASK_ID