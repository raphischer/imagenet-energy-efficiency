from datetime import timedelta
import os
import json
import time
import sys

import tensorflow as tf

from load_imagenet import load_imagenet, resize_with_crop
from util import fix_seed, create_output_dir, Logger, prepare_model, set_gpu


def main(args):
    args.gpu = set_gpu(args.gpu)

    args.seed = fix_seed(args.seed)

    args.output_dir = create_output_dir(os.path.join(args.output_dir, 'train'), args.use_timestamp_dir, args.__dict__)

    # reroute the stdout to logfile, remember to call close!
    sys.stdout = Logger(os.path.join(args.output_dir, 'logfile.txt'))

    dataset = load_imagenet(args.data_path, None, 'train', resize_with_crop, args.batch_size, args.n_batches)
    model = prepare_model(args.model, args.opt.lower(), args.lr, args.momentum, args.weight_decay)

    # TODO Implement custom learning rate scheduling for training RegNet
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

    # removed distributed, mixup transforms & ema code because not used for the selected models

    if args.resume:
        # TODO implement this
        raise NotImplementedError('Resuming training not implemented (yet)')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, 'checkpoint.hdf5'), save_weights_only=True)
    fit_func = lambda: model.fit(dataset, epochs=args.epochs, callbacks=[cp_callback])
    
    # TODO add custom lr scheduler code for tensorflow
    # for epoch in range(args.start_epoch, args.epochs):
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

    start_time = time.time()

    if args.gpu_monitor_interval > 0:
        from gpu_profiling import GpuMonitoringProcess
        monitoring = GpuMonitoringProcess(interval=args.gpu_monitor_interval, outfile=os.path.join(args.output_dir, 'monitoring.json'), gpu_id=args.gpu)
        _, model = monitoring.run(fit_func)
    else:
        model = fit_func()

    print(f"Training finished in {timedelta(seconds=int(time.time() - start_time))} seconds, results can be found in {args.output_dir}")

    if args.gpu_monitor_interval > 0:
        from gpu_profiling import aggregate_log
        print(json.dumps(aggregate_log(os.path.join(args.output_dir, 'monitoring.json')), indent=4))
    
    sys.stdout.close()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--gpu-monitor-interval", default=.5, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--model", default="ResNet50", type=str, help="model name")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
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
    parser.add_argument("--output-dir", default="/raid/fischer/dnns", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")

    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)"),
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--gpu", default=0, type=int, help="gpu to use for computations (if available)")

    return parser


if __name__ == "__main__":
    # TODO custom learning rate scheduling for MobileNet!
    # TODO quantization? quicknet training? larq?
    args = get_args_parser().parse_args()
    main(args)
