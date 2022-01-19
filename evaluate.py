import argparse
from collections import namedtuple
from datetime import timedelta
import json
import os
import time
import sys

import tensorflow as tf

from load_imagenet import load_imagenet
from load_preprocessing import load_preprocessing
from util import fix_seed, create_output_dir, Logger, prepare_model, prepare_optimizer, PatchedJSONEncoder
from monitoring import start_monitoring


def evaluate_single(args):
    args.seed = fix_seed(args.seed)
    custom_trained = os.path.isdir(args.eval_model)

    if custom_trained: # load cfg from training directory
        with open(os.path.join(args.eval_model, 'config.json'), 'r') as m_cfg:
            cfg = json.load(m_cfg)
            # override fields with given args
            cfg.update(args.__dict__)
        args = namedtuple('CFG', cfg)(**cfg)

    else: # prepare to load pretrained weights
        args.model = args.eval_model

    args.output_dir = create_output_dir(os.path.join(args.output_dir, 'eval'), args.use_timestamp_dir, args.__dict__)

    # reroute the stdout to logfile, remember to call close!
    tmp = sys.stdout
    sys.stdout = Logger(os.path.join(args.output_dir, 'logfile.txt'))

    # open strategy scope for using all GPUs
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        if not custom_trained or args.eval_preprocessing == 'builtin':
            preproc_f = load_preprocessing('builtin', args.model, args)
        elif args.eval_preprocessing == 'like-train':
            preproc_f = load_preprocessing(args.preprocessing, args.model, args)
        dataset, ds_info = load_imagenet(args.data_path, None, args.split, preproc_f, args.batch_size, args.n_batches)
        
        if not custom_trained:
            model, _ = prepare_model(args.model, None, weights='pretrained')
        else:
            # TODO check if using default (optimizer = None) makes a difference!
            # currently, this loads the optimizer from the training directory
            optimizer = prepare_optimizer(args.model, args.opt.lower(), args.lr, args.momentum, args.weight_decay, ds_info, args.epochs)
            model, _ = prepare_model(args.model, optimizer, weights=args.eval_model)

    # run monitoring and evaluation
    print("Start evaluation")
    monitoring = start_monitoring(args.gpu_monitor_interval, args.cpu_monitor_interval, args.output_dir)
    start_time = time.time()
    eval_result = model.evaluate(dataset, return_dict=True)
    end_time = time.time()
    for monitor in monitoring:
        monitor.stop()

    # write results
    results = {
        'metrics': eval_result,
        'start': start_time,
        'end': end_time,
        'model': {
            'params': model.count_params()
        }
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as rf:
        json.dump(results, rf, indent=4, cls=PatchedJSONEncoder)

    print(f"Evaluation finished in {timedelta(seconds=int(time.time() - start_time))} seconds, results can be found in {args.output_dir}")

    sys.stdout.close()
    sys.stdout = tmp
    return args.output_dir

    
def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training", add_help=add_help)

    # data and model input
    parser.add_argument("--eval-model", default="/raid/fischer/dnns", type=str, help="path to access the trained model, or name of model to use TF pretrained")
    parser.add_argument("--data-path", default="/raid/imagenet", type=str, help="dataset path")
    parser.add_argument("--split", default="validation", choices=['train', 'validation'], type=str, help="dataset split to use")
    parser.add_argument("--n-batches", default=-1, type=int, help="number of batches to take")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--eval-preprocessing", default='builtin', type=str, help="pass 'builtin' for choosing tf builtin preprocessing according to model choice, or pass a specific model name, or 'like-train' to preprocess like in training")

    # output
    parser.add_argument("--use-timestamp-dir", default=True, action="store_true", help="Creates timestamp directory in data path")
    parser.add_argument("--output-dir", default="/raid/fischer/eval", type=str, help="path to save outputs")
    parser.add_argument("--gpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates GPU profiling every X seconds")
    parser.add_argument("--cpu-monitor-interval", default=1, type=float, help="Setting to > 0 activates CPU profiling every X seconds")

    # randomization and hardware
    parser.add_argument("--seed", type=int, default=-1, help="Seed to use (if -1, uses and logs random seed)")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    evaluate_single(args)
