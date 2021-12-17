from datetime import timedelta
import os
import json
import time
import sys

import tensorflow as tf

from load_imagenet import load_imagenet, resize_with_crop
from util import fix_seed, create_output_dir, Logger, prepare_model, set_gpu, prepare_optimizer, start_monitoring
from util import prepare_lr_scheduling


def main(args):
    args.gpu = set_gpu(args.gpu)

    args.seed = fix_seed(args.seed)

    args.output_dir = create_output_dir(os.path.join(args.output_dir, 'train'), args.use_timestamp_dir, args.__dict__)

    # reroute the stdout to logfile, remember to call close!
    sys.stdout = Logger(os.path.join(args.output_dir, 'logfile.txt'))

    dataset, ds_info = load_imagenet(args.data_path, None, 'train', resize_with_crop, args.batch_size, args.n_batches)
    optimizer = prepare_optimizer(args.model, args.opt.lower(), args.lr, args.momentum, args.weight_decay, ds_info, args.epochs)
    model = prepare_model(args.model, optimizer)

    save_freq = ds_info.steps_per_epoch * 10 # every 10 epochs
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, 'checkpoint_{epoch:03d}.hdf5'), save_weights_only=True, save_freq=save_freq)]

    lr_callback = prepare_lr_scheduling(args.lr_scheduler, args.lr_gamma, args.lr_step_size, args.lr)
    if lr_callback is not None:
        callbacks.append(lr_callback)

    # removed distributed, mixup transforms & ema code because not used for the selected models

    if args.resume:
        # TODO implement this
        raise NotImplementedError('Resuming training not implemented (yet)')

    start_time = time.time()

    monitoring = start_monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, args.output_dir, args.gpu)

    model.fit(dataset, epochs=args.epochs, callbacks=callbacks)

    for monitor in monitoring:
        monitor.stop()

    print(f"Training finished in {timedelta(seconds=int(time.time() - start_time))} seconds, results can be found in {args.output_dir}")

    sys.stdout.close()

    return args.output_dir

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


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--gpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--model", default="QuickNet", type=str, help="model name")
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
    # TODO implement custom learning rate scheduling for MobileNet!
    # TODO quantization?
    args = get_args_parser().parse_args()
    main(args)
    from monitoring import aggregate_log
    gpu_log = aggregate_log(os.path.join(args.output_dir, 'monitoring_gpu.json'))
    cpu_log = aggregate_log(os.path.join(args.output_dir, 'monitoring_cpu.json'))
