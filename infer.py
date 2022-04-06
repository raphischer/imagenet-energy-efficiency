import argparse
from datetime import timedelta
import importlib
import json
import os
import time
import sys

from mlee.util import fix_seed, create_output_dir, Logger, PatchedJSONEncoder
from mlee.monitoring import Monitoring


def evaluate_single(args):
    args.seed = fix_seed(args.seed)
    if os.path.basename(args.infer_model) == '': # reformat paths ending with / behind the directory
        setattr(args, 'infer_model', os.path.dirname(args.infer_model))
    custom_trained = os.path.isdir(args.infer_model)

    if custom_trained: # load cfg from training directory
        with open(os.path.join(args.infer_model, 'config.json'), 'r') as m_cfg:
            cfg = json.load(m_cfg)
            # override fields with given args, except for backend which has to align with trained model
            backend = args.backend
            cfg.update(args.__dict__)
            cfg['backend'] = backend
        for key, val in cfg.items():
            setattr(args, key, val)

    else: # prepare to load pretrained weights
        args.model = args.infer_model

    args.output_dir = create_output_dir(os.path.join(args.output_dir, os.path.basename(__file__)[:-3]), args.__dict__)
    try:
        backend = importlib.import_module(f'mlee.ml_{args.backend}.infer')
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f'Error when loading backend {args.backend}!\n  {e}')

    for split in ['validation']:

        # reroute the stdout to logfile, remember to call close!
        tmp = sys.stdout
        sys.stdout = Logger(os.path.join(args.output_dir, f'{split}_logfile.txt'))

        eval_func, model_info = backend.init_inference(args, split)

        # run monitoring and evaluation
        print("Start evaluation")
        monitoring = Monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, args.output_dir, f'{split}_')
        start_time = time.time()
        eval_result = eval_func()
        end_time = time.time()
        monitoring.stop()

        # write results
        results = {
            'metrics': backend.finalize_inference(eval_result),
            'start': start_time,
            'end': end_time,
            'model': model_info
        }
        with open(os.path.join(args.output_dir, f'{split}_results.json'), 'w') as rf:
            json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

        print(f"Evaluation finished in {timedelta(seconds=int(time.time() - start_time))} seconds, results can be found in {args.output_dir}")

        sys.stdout.close()
        sys.stdout = tmp
    return args.output_dir

    
def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    # data and model input
    parser.add_argument("--infer-model", default="/raid/tmpler/dnns", type=str, help="path to training directory, or name of pretrained model")
    parser.add_argument("--backend", default="tensorflow", type=str, choices=["tensorflow", "pytorch", "onnx_pytorch", "onnx_tensorflow"], help="machine learning software to use")
    parser.add_argument("--data-path", default="/raid/imagenet_tensorflow", type=str, help="dataset path")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--eval-preprocessing", default='builtin', type=str, help="pass 'builtin' for choosing tf builtin preprocessing according to model choice, or pass a specific model name, or 'like-train' to preprocess like in training")

    # output
    parser.add_argument("--output-dir", default="/raid/tmpler/eval", type=str, help="path to save outputs")
    parser.add_argument("--gpu-monitor-interval", default=.2, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=.2, type=float, help="Setting to > 0 activates CPU profiling every X seconds")

    # randomization and hardware
    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    evaluate_single(args)
