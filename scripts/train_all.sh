#!/bin/sh

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

for model in EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3 EfficientNetB4
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model --opt rmsprop --weight-decay 0.00001 --lr 0.256 --lr-step-size 2.4 --lr-gamma 0.97
done

if [ $2 == "tensorflow" ]
then
for model in DenseNet121 DenseNet169 DenseNet201 InceptionResNetV2 Xception NASNetMobile QuickNet QuickNetSmall QuickNetLarge
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model # --epochs 600
done
fi

if [ $2 == "pytorch" ]
then
for model in ResNext50 ResNext101 RegNetX400MF RegNetX8GF RegNetX32GF
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model
rm -rf ~/.cache/torch/hub/checkpoints
done
fi

for model in EfficientNetB5 EfficientNetB6 EfficientNetB7:
do
python train.py --backend $2 --data-path $3 --output-dir $1 --model $model --opt rmsprop --weight-decay 0.00001 --lr 0.256 --lr-step-size 2.4 --lr-gamma 0.97
done