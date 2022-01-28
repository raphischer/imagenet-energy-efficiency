#!/bin/bash
OUT_DIR=raid

python train.py --model VGG16 --output-dir $OUT_DIR/vgg16 --lr 1e-2 --data-path raid/ImageNet
python train.py --model VGG19 --output-dir $OUT_DIR/vgg19 --lr 1e-2 --data-path raid/ImageNet