from datetime import timedelta
import os
import json
import time
import sys

import tensorflow as tf

from load_imagenet import load_imagenet
from load_preprocessing import load_preprocessing
from monitoring import start_monitoring
from util import fix_seed, create_output_dir, Logger, prepare_model, prepare_optimizer, prepare_lr_scheduling, PatchedJSONEncoder, TimestampOnEpochEnd


def main(args):
    args.seed = fix_seed(args.seed)
    args.output_dir = create_output_dir(os.path.join(args.output_dir, 'train'), args.use_timestamp_dir, args.__dict__)

    # reroute the stdout to logfile, remember to call close!
    tmp = sys.stdout
    sys.stdout = Logger(os.path.join(args.output_dir, 'logfile.txt'))

    # open strategy scope for using all GPUs
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        preproc_f = load_preprocessing(args.preprocessing, args.model, args)
        dataset, ds_info = load_imagenet(args.data_path, None, 'train', preproc_f, args.batch_size, args.n_batches)
        optimizer = prepare_optimizer(args.model, args.opt.lower(), args.lr, args.momentum, args.weight_decay, ds_info, args.epochs)

        if args.resume:
            model, mfile = prepare_model(args.model, optimizer, weights=args.resume)
            initial_epoch = int(mfile[11:14])
        else:
            model, _ = prepare_model(args.model, optimizer)
            initial_epoch = 0

    # create callbacks
    callbacks = [TimestampOnEpochEnd(os.path.join(args.output_dir, "epoch_timestamps.csv"))]
    for i in [10, 5, 2, 1]:
        if args.epochs % i == 0:
            save_freq = ds_info.steps_per_epoch * i # checkpoints every i epochs
            break
    callbacks.append([tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, 'checkpoint_{epoch:03d}.hdf5'), save_weights_only=True, save_freq=save_freq)])
    lr_callback = prepare_lr_scheduling(args.lr_scheduler, args.lr_gamma, args.lr_step_size, args.lr)
    if lr_callback is not None:
        callbacks.append(lr_callback)

    # start monitoring and train
    monitoring = start_monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, args.output_dir)
    start_time = time.time()
    res = model.fit(dataset, epochs=args.epochs, callbacks=callbacks, initial_epoch=initial_epoch)
    end_time = time.time()
    for monitor in monitoring:
        monitor.stop()

    # write results
    results = {
        'history': res.history,
        'start': start_time,
        'end': end_time,
        'model': {
            'params': res.model.count_params(),
            'fsize': os.path.getsize(os.path.join(args.output_dir, f'checkpoint_{args.epochs:03d}.hdf5'))
        }
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

    print(f"Training finished in {timedelta(seconds=int(end_time - start_time))} seconds, results can be found in {args.output_dir}")

    sys.stdout.close()
    sys.stdout = tmp
    return args.output_dir


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    # model and data input
    parser.add_argument("--model", default="ResNet50", type=str, help="model name")
    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch will be $NGPU x batch_size")

    # output
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--output-dir", default="/raid/fischer/dnns", type=str, help="path to save outputs")
    parser.add_argument("--gpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates CPU profiling every X seconds")

    # training parameters
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

    # data preprocessing
    parser.add_argument("--preprocessing", default='builtin', type=str, help="pass 'builtin' for choosing tf builtin preprocessing according to model choice, or pass a specific model name, or 'custom' with using the parameters below")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")

    # randomization and hardware
    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)"),

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
