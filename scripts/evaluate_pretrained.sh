#!/bin/sh

export PYTHONHASHSEED=0

for split in validation train
do
# KERAS:
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model VGG16,VGG19
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model ResNet50,ResNet101,ResNet152
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model ResNet50V2,ResNet101V2,ResNet152V2
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model MobileNetV2,MobileNetV3Small,MobileNetV3Large
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model DenseNet121,DenseNet169,DenseNet201
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model InceptionResNetV2,InceptionV3
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model Xception,NASNetMobile # ,NASNetLarge
python evaluate_all.py --output-dir $1 --split $split --eval-model EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
# LARQ:
# python evaluate_all.py --data-path /home/fischer/mnt_imagenet --output-dir $1 --split $split --eval-model QuickNet,QuickNetSmall,QuickNetLarge
done
