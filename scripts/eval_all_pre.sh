#!/bin/sh

### default parameters taken from https://github.com/pytorch/vision/tree/main/references/classification

export PYTHONHASHSEED=0

for model in VGG16 VGG19
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
done

for model in ResNet50 ResNet101 ResNet152
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
done

python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model MobileNetV2
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model MobileNetV3Small
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model MobileNetV3Large

# hyperparameters taken from https://arxiv.org/pdf/1905.11946.pdf, what is missing is batch norm momentum 0.99, SiLU activation, AutoAugment, and stochastic depth
for model in EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3 EfficientNetB4 # EfficientNetB5 EfficientNetB6 EfficientNetB7:
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
done

### QuickNet parameters parsed from https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py, with the custom opimizer implemented
# python evaluate.py output-dir $1 --eval-model QuickNet # --epochs 600
# python evaluate.py --eval-model QuickNetSmall # --epochs 600
# python evaluate.py --eval-model QuickNetLarge # --epochs 600

# ResNext?

# EfficientNet?

# RegNet?

# Quantized?
