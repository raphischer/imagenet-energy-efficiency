### inspired by https://github.com/pytorch/vision/tree/main/references/classification

import datetime
import os
import time
import warnings

import transforms

import tensorflow as tf

from load_imagenet import load_imagenet, resize_with_crop

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


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
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir)

    print(args)

    if args.use_deterministic_algorithms:
        pass # TODO elaborate?

    raw_dir = os.path.join(args.data_path, "raw")
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    dataset_train = load_imagenet(raw_dir, train_dir, 'train', resize_with_crop, args.batch_size)
    # dataset_test = load_imagenet(raw_dir, train_dir, 'test', resize_with_crop, args.batch_size)

    # removed mixup transforms because not used on webpage model commands

    model = tf.keras.applications.__dict__[args.model](include_top=True, weights=None)

    criterion = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=args.lr,
            momentum=args.momentum,
            decay=args.weight_decay,
            nesterov="nesterov" in opt_name
        )
    elif opt_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(args.lr, rho=0.9, momentum=args.momentum, epsilon=0.0316, decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and RMSprop are supported.")

    # Only used for RegNet
    # args.lr_scheduler = args.lr_scheduler.lower()
    # if args.lr_scheduler == "steplr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # elif args.lr_scheduler == "cosineannealinglr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=args.epochs - args.lr_warmup_epochs
    #     )
    # elif args.lr_scheduler == "exponentiallr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    # else:
    #     raise RuntimeError(
    #         f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
    #         "are supported."
    #     )

    # if args.lr_warmup_epochs > 0:
    #     if args.lr_warmup_method == "linear":
    #         warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #             optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
    #         )
    #     elif args.lr_warmup_method == "constant":
    #         warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
    #             optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
    #         )
    #     else:
    #         raise RuntimeError(
    #             f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
    #         )
    #     lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
    #         optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
    #     )
    # else:
    #     lr_scheduler = main_lr_scheduler

    # removed distributed code

    model_ema = None # code removed because not used on Torch webpage

    if args.resume:
        # TODO implement this
        raise NotImplementedError('Resuming training not implemented (yet)')
        # checkpoint = torch.load(args.resume, map_location="cpu")
        # model_without_ddp.load_state_dict(checkpoint["model"])
        # if not args.test_only:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # args.start_epoch = checkpoint["epoch"] + 1
        # if model_ema:
        #     model_ema.load_state_dict(checkpoint["model_ema"])
        # if scaler:
        #     scaler.load_state_dict(checkpoint["scaler"])

    print("Start training")
    start_time = time.time()

    metrics = ['accuracy']    
    model.trainable = True
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.output_dir,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.fit(dataset_train, epochs=args.epochs, callbacks=[cp_callback])

    # TODO add custom lr scheduler code for tensorflow
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #     train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
    #     lr_scheduler.step()
    #     evaluate(model, criterion, data_loader_test, device=device)
    #     if model_ema:
    #         evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
    #     if args.output_dir:
    #         checkpoint = {
    #             "model": model_without_ddp.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "lr_scheduler": lr_scheduler.state_dict(),
    #             "epoch": epoch,
    #             "args": args,
    #         }
    #         if model_ema:
    #             checkpoint["model_ema"] = model_ema.state_dict()
    #         if scaler:
    #             checkpoint["scaler"] = scaler.state_dict()
    #         utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
    #         utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
