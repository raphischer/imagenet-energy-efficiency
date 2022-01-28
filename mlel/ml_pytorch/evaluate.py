import os
import warnings

import torch
from torch import nn
import torch.utils.data
import torchvision
import mlel.ml_pytorch.pt_presets as presets
import mlel.ml_pytorch.pt_utils as utils

from torchvision.transforms.functional import InterpolationMode

model_name_mapping = {
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "ResNet50": "resnet50",
    "ResNet101": "resnet101",
    "ResNet152": "resnet152",
    "MobileNetV2": "mobilenet_v2",
    "MobileNetV3Large": "mobilenet_v3_large",
    "MobileNetV3Small": "mobilenet_v3_small",
    "EfficientNetB0": "efficientnet_b0",
    "EfficientNetB1": "efficientnet_b1",
    "EfficientNetB2": "efficientnet_b2",
    "EfficientNetB3": "efficientnet_b3",
    "EfficientNetB4": "efficientnet_b4",
    "EfficientNetB5": "efficientnet_b5",
    "EfficientNetB6": "efficientnet_b6",
    "EfficientNetB7": "efficientnet_b7",
    "InceptionV3": "inception_v3"
}

def _evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", return_dict=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    loss = metric_logger.loss.global_avg
    acc1 = metric_logger.acc1.global_avg
    acc5 = metric_logger.acc5.global_avg

    if return_dict:
        return {"loss": loss, "accuracy": acc1/100, "top_5_accuracy": acc5/100}
    else:
        return loss, acc1, acc5

def load_data(traindir, valdir, args):
    # Data loading code
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
    )

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def init_evaluation(args, split):
    torch.manual_seed(args.seed)
    custom_trained = os.path.isdir(args.eval_model)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set missing args, depending on model name
    args.interpolation = "bilinear"
    args.val_resize_size = 224
    args.val_crop_size = 224
    args.train_crop_size = 224

    if "EfficientNet" in args.model:
        args.interpolation = "bicubic"
    if args.model == "EfficientNetB0":
        args.val_resize_size = 256
        args.val_crop_size = 224
        args.train_crop_size = 224
    if args.model == "EfficientNetB1":
        args.val_resize_size = 256
        args.val_crop_size = 240
        args.train_crop_size = 240
    if args.model == "EfficientNetB2":
        args.val_resize_size = 288
        args.val_crop_size = 288
        args.train_crop_size = 288
    if args.model == "EfficientNetB3":
        args.val_resize_size = 320
        args.val_crop_size = 300
        args.train_crop_size = 300
    if args.model == "EfficientNetB4":
        args.val_resize_size = 384
        args.val_crop_size = 380
        args.train_crop_size = 380
    if args.model == "EfficientNetB5":
        args.val_resize_size = 456
        args.val_crop_size = 456
        args.train_crop_size = 456
    if args.model == "EfficientNetB6":
        args.val_resize_size = 528
        args.val_crop_size = 528
        args.train_crop_size = 528
    if args.model == "EfficientNetB7":
        args.val_resize_size = 600
        args.val_crop_size = 600
        args.train_crop_size = 600

    if args.model == "InceptionV3":
        args.val_resize_size = 342
        args.val_crop_size = 299
        args.train_crop_size = 299

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Load data
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    num_classes = len(dataset_train.classes)

    if split == "train":
        data_loader_test = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=16, pin_memory=True)
    else:
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)


    # Create model
    if custom_trained:
        # Load weights from folder
        model = torchvision.models.__dict__[model_name_mapping[args.model]](pretrained=False, num_classes=num_classes)

        last_model =  sorted([f for f in os.listdir(args.eval_model) if f.startswith('checkpoint')])[-1]
        model.load_state_dict(torch.load(os.path.join(args.eval_model, last_model)))
    else:
        # Use pretrained weights
        model = torchvision.models.__dict__[model_name_mapping[args.model]](pretrained=True, num_classes=num_classes)

    torch.save(model.state_dict(), os.path.join(args.output_dir, f"eval_weights.pth"))
    model = nn.DataParallel(model)
    model.to(device)

    model_info = {
        'params': sum(p.numel() for p in model.parameters()), 
        'fsize': os.path.getsize(os.path.join(args.output_dir, f"eval_weights.pth"))
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    eval_func = lambda: _evaluate(model, criterion, data_loader_test, device, return_dict=True)
    return eval_func, model_info

def finalize_evaluation(results):
    return results
