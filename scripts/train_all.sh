#!/bin/sh

### default parameters taken from https://github.com/pytorch/vision/tree/main/references/classification

export PYTHONHASHSEED=0

# python3 train.py --output-dir $1 --model VGG16 --lr 1e-2 # --epochs 90
python3 train.py --model VGG19 --lr 1e-2 # --epochs 90

# python3 train.py --output-dir $1 --model ResNet50 # --epochs 90
python3 train.py --model ResNet101 # --epochs 90
python3 train.py --model ResNet152 # --epochs 90

# python3 train.py --model MobileNetV2 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 # --epochs 300
# python3 train.py --output-dir $1 --model MobileNetV3Small --opt rmsprop --lr 0.064 --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2 # --epochs 600
# python3 train.py --model MobileNetV3Large --opt rmsprop --lr 0.064 --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2 # --epochs 600

# hyperparameters taken from https://arxiv.org/pdf/1905.11946.pdf, what is missing is batch norm momentum 0.99, SiLU activation, AutoAugment, and stochastic depth
# for model in EfficientNetB0 EfficientNetB1: # EfficientNetB2 EfficientNetB3 EfficientNetB4 EfficientNetB5 EfficientNetB6 EfficientNetB7:
# do
# python3 train.py --output-dir $1 --model $model --opt rmsprop --wd 0.00001 --lr 0.256 --lr-step-size 2.4 --lr-gamma 0.97
# done

### QuickNet parameters parsed from https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py, with the custom opimizer implemented
python3 train.py  --output-dir $1 --model QuickNet # --epochs 600
# python3 train.py --model QuickNetSmall # --epochs 600
# python3 train.py --model QuickNetLarge # --epochs 600

# ResNext?

# EfficientNet?

# RegNet?

# Quantized?
