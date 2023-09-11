#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=urop
#SBATCH --output=../sbatches/logs/debug/ccs.out
#SBATCH --error=../sbatches/logs/debug/ccs.err

python ../generate.py --model_name deberta --dataset imdb --num_examples 400 --batch_size 40