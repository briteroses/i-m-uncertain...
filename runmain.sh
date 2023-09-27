#!/bin/bash

# Specify the GPU to use
export CUDA_VISIBLE_DEVICES=14

# Activate conda environment
source /home/joe.kwon/anaconda3/etc/profile.d/conda.sh
conda activate uncertainty

# Run Python script and save logs to a file
#python main.py --uncertainty True 

python main.py --uncertainty True > output.log 2>&1
