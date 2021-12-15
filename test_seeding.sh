#!/bin/sh

export PYTHONHASHSEED=0
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

# when setting the following variable, code is not executed on GPU which provides deterministic behaviour
# export CUDA_VISIBLE_DEVICES=-1

/bin/python3 train.py --model VGG16 --lr 1e-2 --epochs 1 --n-batches 50 --batch-size 32 --gpu-monitor-interval -1 --seed 0
