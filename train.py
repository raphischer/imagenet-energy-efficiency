from datetime import timedelta
import os
import json
import time
import sys
import importlib

from mlee.monitoring import Monitoring
from mlee.util import fix_seed, create_output_dir, Logger, PatchedJSONEncoder


def main(args):
    args.seed = fix_seed(args.seed)
    args.output_dir = create_output_dir(os.path.join(args.output_dir, os.path.basename(__file__)[:-3]), args.__dict__)

    # reroute the stdout to logfile, remember to call close!
    tmp = sys.stdout
    sys.stdout = Logger(os.path.join(args.output_dir, 'logfile.txt'))
    
    try:
        backend = importlib.import_module(f'mlee.ml_{args.backend}.train')
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f'Error when loading backend {args.backend}!\n  {e}')

    fit_func = backend.init_training(args)

    # start monitoring and train
    monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, args.output_dir)
    start_time = time.time()
    train_result = fit_func()
    end_time = time.time()
    monitoring.stop()

    results = {'start': start_time, 'end': end_time}
    results = backend.finalize_training(train_result, results, args)

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
    parser.add_argument("--backend", default="tensorflow", type=str, choices=["tensorflow", "pytorch"], help="machine learning software to use")
    parser.add_argument("--model", default="ResNet50", type=str, help="model name")
    parser.add_argument("--data-path", default="/raid/imagenet_tensorflow", type=str, help="dataset path")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("--batch-size", default=32, type=int, help="images per gpu, the total batch will be $NGPU x batch_size")

    # output & experiment settings
    parser.add_argument("--output-dir", default="/raid/tmpler/dnns", type=str, help="path to save outputs")
    parser.add_argument("--gpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates CPU profiling every X seconds")
    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)"),

    # training parameters
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--early-patience", default=50, type=int, help="early stopping patience")
    parser.add_argument("--early-delta", default=0.01, type=float, help="early stopping min delta")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--opt-decy", default=0.9, type=str, help="discounting factor rho for rmsprop")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-step-size", default=30, type=float, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")

    # data preprocessing
    parser.add_argument("--preprocessing", default='builtin', type=str, help="pass 'builtin' for choosing builtin preprocessing according to model choice, or pass a specific model name, or 'custom' with using the parameters below")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
