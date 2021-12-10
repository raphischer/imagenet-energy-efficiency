### inspired by https://github.com/pytorch/vision/tree/main/references/classification

import datetime
import os
import json
import time

import tensorflow as tf

from load_imagenet import load_imagenet, resize_with_crop


def main(args):

    # create log dir
    timestamp = datetime.datetime.now().strftime('train_%Y_%m_%d_%H_%M')
    if args.output_dir:
        if args.use_timestamp_dir:
            args.output_dir = os.path.join(args.output_dir, timestamp)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    # write args
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as cfg:
        d_args = args.__dict__
        d_args['timestamp'] = timestamp
        json.dump(d_args, cfg, indent=4)

    if args.use_deterministic_algorithms:
        pass # TODO elaborate?

    dataset_train = load_imagenet(args.data_path, None, 'train', resize_with_crop, args.batch_size)
    if args.n_batches > 0:
        dataset_train = dataset_train.take(args.n_batches)

    # removed mixup transforms because not used on webpage model commands

    try:
        model = tf.keras.applications.__dict__[args.model](include_top=True, weights=None)
    except (TypeError, KeyError) as e:
        avail = ', '.join(n for n, e in tf.keras.applications.__dict__.items() if callable(e))
        raise RuntimeError(f'Error when loading {args.model}! \n{e}\nAvailable models:\n{avail}')

    # TODO which Crossentropy to use?!
    # criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

    # TODO Implement for training RegNet
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

    # removed ema code because not used on Torch webpage

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

    metrics = ['accuracy']    
    model.trainable = True
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, args.model.lower()),
                                                     save_weights_only=True,
                                                     verbose=1)

    fit_model = lambda: model.fit(dataset_train, epochs=args.epochs, callbacks=[cp_callback])
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
    
    print("Start training")
    start_time = time.time()

    if args.gpu_monitor_interval > 0:
        from gpu_profiling import GpuMonitoringProcess
        monitoring = GpuMonitoringProcess(interval=args.gpu_monitor_interval)
        monitoring_result, model = monitoring.run(fit_model)
    else:
        model = fit_model()
        monitoring_result = {}

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    with open(os.path.join(args.output_dir, 'monitoring.json'), 'w') as cfg:
        json.dump(monitoring_result, cfg, indent=4)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--gpu-monitor-interval", default=-1, type=int, help="Setting to > 0 activates GPU profiling")
    parser.add_argument("--model", default="ResNet50", type=str, help="model name")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="/raid/fischer/checkpoints", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
