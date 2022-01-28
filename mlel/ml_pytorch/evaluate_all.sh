for split in validation train
do
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model VGG16 --output-dir /raid/fischer/torch_eval_pre/VGG16$split
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model VGG19 --output-dir /raid/fischer/torch_eval_pre/VGG19$split
rm -rf ~/.cache/torch/hub/checkpoints

torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model ResNet50 --output-dir /raid/fischer/torch_eval_pre/ResNet50$split
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model ResNet101 --output-dir /raid/fischer/torch_eval_pre/ResNet101$split
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model ResNet152 --output-dir /raid/fischer/torch_eval_pre/ResNet152$split
rm -rf ~/.cache/torch/hub/checkpoints

torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model MobileNetV2 --output-dir /raid/fischer/torch_eval_pre/MobileNetV2$split
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model MobileNetV3Large --output-dir /raid/fischer/torch_eval_pre/MobileNetV3Large$split
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model MobileNetV3Small --output-dir /raid/fischer/torch_eval_pre/MobileNetV3Small$split
rm -rf ~/.cache/torch/hub/checkpoints

torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB0 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB0$split --interpolation bicubic --val-resize-size 256 --val-crop-size 224 --train-crop-size 224
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB1 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB1$split --interpolation bicubic --val-resize-size 256 --val-crop-size 240 --train-crop-size 240
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB2 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB2$split --interpolation bicubic --val-resize-size 288 --val-crop-size 288 --train-crop-size 288
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB3 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB3$split --interpolation bicubic --val-resize-size 320 --val-crop-size 300 --train-crop-size 300
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB4 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB4$split --interpolation bicubic --val-resize-size 384 --val-crop-size 380 --train-crop-size 380
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB5 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB5$split --interpolation bicubic --val-resize-size 456 --val-crop-size 456 --train-crop-size 456
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB6 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB6$split --interpolation bicubic --val-resize-size 528 --val-crop-size 528 --train-crop-size 528
rm -rf ~/.cache/torch/hub/checkpoints
torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model EfficientNetB7 --output-dir /raid/fischer/torch_eval_pre/EfficientNetB7$split --interpolation bicubic --val-resize-size 600 --val-crop-size 600 --train-crop-size 600
rm -rf ~/.cache/torch/hub/checkpoints

torchrun --nproc_per_node=8 train.py --data-path /raid/imagenet_torch/ImageNet --pretrained --test-only --split $split --model InceptionV3 --output-dir /raid/fischer/torch_eval_pre/InceptionV3$split --val-resize-size 342 --val-crop-size 299 --train-crop-size 299
rm -rf ~/.cache/torch/hub/checkpoints

done
