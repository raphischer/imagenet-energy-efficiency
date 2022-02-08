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

for model in MobileNetV2 MobileNetV3Small MobileNetV3Large
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
done

for model in EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3 EfficientNetB4 EfficientNetB5 EfficientNetB6 EfficientNetB7
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
done

python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model InceptionV3

#  Available For Tensorflow Only
if [ $2 == "tensorflow" ]
then
for model in DenseNet121 DenseNet169 DenseNet201 InceptionResNetV2 Xception NASNetMobile QuickNet QuickNetSmall QuickNetLarge
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
done
fi

#  Available For Pytorch Only
if [ $2 == "pytorch" ]
then
for model in ResNext50 ResNext101 RegNetX400MF RegNetX8GF RegNetX32GF
do
python evaluate.py --backend $2 --data-path $3_$2 --output-dir $1 --eval-model $model
rm -rf ~/.cache/torch/hub/checkpoints
done
fi