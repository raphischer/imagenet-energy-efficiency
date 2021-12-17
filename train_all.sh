#!/bin/sh

### default parameters taken from https://github.com/pytorch/vision/tree/main/references/classification

export PYTHONHASHSEED=0

python3 train.py --model VGG16 --lr 1e-2
python3 train.py --model VGG19 --lr 1e-2

python3 train.py --model ResNet50
python3 train.py --model ResNet101
python3 train.py --model ResNet152 --batch-size 128 # 256 too much for A100

python3 train.py --model MobileNetV2 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98
python3 train.py --model MobileNetV3Small --opt rmsprop --lr 0.064 --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2
python3 train.py --model MobileNetV3Large --opt rmsprop --lr 0.064 --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2

### QuickNet parameters parsed from https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py

python3 train.py --model QuickNet --epochs 600 --lr-scheduler None
python3 train.py --model QuickNetSmall --epochs 600 --lr-scheduler None
python3 train.py --model QuickNetLarge --epochs 600 --lr-scheduler None

# ResNext?

# EfficientNet?

# RegNet?

# Quantized?
