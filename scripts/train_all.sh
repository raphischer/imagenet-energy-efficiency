#!/bin/sh

### default parameters taken from https://github.com/pytorch/vision/tree/main/references/classification

export PYTHONHASHSEED=0

for model in VGG16 VGG19
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model --lr 1e-2 # --epochs 90
done

for model in ResNet50 ResNet101 ResNet152
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model # --epochs 90
done

python train.py --backend $2 --data-path $3 --output-dir $1 --model MobileNetV2 --lr 0.045 --weight-decay 0.00004 --lr-step-size 1 --lr-gamma 0.98 # --epochs 300
python train.py --backend $2 --data-path $3 --output-dir $1 --model MobileNetV3Small --opt rmsprop --lr 0.064 --weight-decay 0.00001 --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2 # --epochs 600
python train.py --backend $2 --data-path $3 --output-dir $1 --model MobileNetV3Large --opt rmsprop --lr 0.064 --weight-decay 0.00001 --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2 # --epochs 600

# hyperparameters taken from https://arxiv.org/pdf/1905.11946.pdf, what is missing is batch norm momentum 0.99, SiLU activation, AutoAugment, and stochastic depth
for model in EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3 EfficientNetB4 # EfficientNetB5 EfficientNetB6 EfficientNetB7:
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model --opt rmsprop --weight-decay 0.00001 --lr 0.256 --lr-step-size 2.4 --lr-gamma 0.97
done

if [ $2 == "tensorflow" ]
then
for model in DenseNet121 DenseNet169 DenseNet201 InceptionResNetV2 Xception NASNetMobile QuickNet QuickNetSmall QuickNetLarge
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model
done
fi

## QuickNet parameters parsed from https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py, with the custom opimizer implemented
python train.py output-dir $1 --model QuickNet # --epochs 600
python train.py --model QuickNetSmall # --epochs 600
python train.py --model QuickNetLarge # --epochs 600
