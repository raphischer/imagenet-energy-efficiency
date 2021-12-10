#!/bin/sh

# AlexNet and VGG
for MODEL in AlexNet VGG11 VGG13 VGG16 VGG19
do
  python train.py --model $MODEL --lr 1e-2
done

# ResNet
for MODEL in ResNet18 ResNet34 ResNet50 ResNet101 ResNet152
do
  train.py --model $MODEL
done

# MobileNetV2
train.py --model MobileNetV2 --epochs 300 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98

# MobileNetV3 Large & Small
for MODEL in MobileNetV3Large MobileNetV3Small
do
  train.py --model $MODEL --epochs 600 --opt rmsprop --batch-size 128 --lr 0.064\ 
     --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2
done

# ResNext?

# EfficientNet?

# RegNet?

# Quantized?