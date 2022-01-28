import os
import time
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

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=False):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return metric_logger.loss.global_avg,metric_logger.acc1.global_avg,metric_logger.acc5.global_avg,


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
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

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg

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

def _train(data_loader, data_loader_test, device, model, criterion, optimizer, lr_scheduler, args):

    history = {
        "timestamp": [],
        "loss": [],
        "accuracy": [],
        "top_5_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_top_5_accuracy": [],
        "lr": [],
    }

    for epoch in range(args.epochs):

        history["timestamp"].append(time.time() * 1000)

        history["lr"].append(0)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc1 / 100)
        history["top_5_accuracy"].append(train_acc5 / 100)

        lr_scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate(model, criterion, data_loader_test, device=device)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc1 / 100)
        history["val_top_5_accuracy"].append(val_acc5 / 100)

        torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_{epoch:03d}.pth"))

    # results structure from other project
    results = {
        "history": history,
        "model": model,

    }
    return results

def finalize_training(train_res, results, args):
    final_epoch = len(train_res["history"]["loss"])
    torch.save(train_res["model"].state_dict(), os.path.join(args.output_dir, f"checkpoint_{final_epoch:03d}_final.pth"))

    results.update({
        'history': train_res["history"],
        'model': {
            'params': sum(p.numel() for p in train_res["model"].parameters()), # TODO: This is number of all parameters, even untrainable parameters,
            'fsize': os.path.getsize(os.path.join(args.output_dir, f'checkpoint_{final_epoch:03d}_final.pth'))
        }
    })
    return results


def init_training(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        setattr(args, "batch_size", args.batch_size * torch.cuda.device_count())

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
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)

    # Create model and training parameters
    model = torchvision.models.__dict__[model_name_mapping[args.model]](pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    parameters = model.parameters()

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    lr_scheduler = main_lr_scheduler
    return lambda: _train(data_loader, data_loader_test, device, model, criterion, optimizer, lr_scheduler, args)
