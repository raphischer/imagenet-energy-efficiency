import argparse
from copy import deepcopy
import os
import subprocess


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    # data and model input
    parser.add_argument("--model-dir", default="/raid/fischer/dnns", type=str, help="path to access the trained model (can also be directory with multiple model subdirs)")
    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--split", default="validation", choices=['train', 'validation'], type=str, help="dataset split to use")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--always-use-simple-prep", default=False, action="store_true", help="")
    parser.add_argument("--pretrained", default=False, action="store_true", help="")

    # output
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--output-dir", default="/raid/fischer/eval", type=str, help="path to save outputs")
    parser.add_argument("--gpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates CPU profiling every X seconds")

    # randomization and hardware
    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)"),
    parser.add_argument("--gpu", default=0, type=int, help="gpu to use for computations (if available)")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    root_dir = args.model_dir
    for subdir in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, subdir, 'monitoring_gpu.json')): # only if training finished
            args_copy = deepcopy(args)
            args_copy.model_dir = os.path.join(root_dir, subdir)
            run_args = ["python", "evaluate.py"]
            for key, val in args_copy.__dict__.items():
                if isinstance(val, bool):
                    if val:
                        run_args.append('--' + key.replace('_', '-'))
                else:
                    run_args.extend(['--' + key.replace('_', '-'), str(val)])
            subprocess.run(run_args)
