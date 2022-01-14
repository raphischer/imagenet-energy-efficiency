from datetime import datetime
from genericpath import isfile
import json
import math
import os
import random as python_random
import sys

import tensorflow as tf
import larq_zoo as lqz
from larq_zoo.training.learning_schedules import CosineDecayWithWarmup
import larq as lq
import numpy as np


TF_MODELS = {n: e for n, e in tf.keras.applications.__dict__.items() if callable(e)}
QUICKNETS = {n: e for n, e in lqz.sota.__dict__.items() if callable(e)}
MODELS = {**TF_MODELS, **QUICKNETS}


class PatchedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def set_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > gpu_id:
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
    # GPU usage might be limited by environment variable
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        limited_gpus = [int(g) for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpu_id = limited_gpus[gpu_id]
    return gpu_id


def fix_seed(seed):
    if seed == -1:
        seed = python_random.randint(0, 2**32 - 1)

    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    return seed


def prepare_optimizer(model_name, opt_name, lr, momentum, weight_decay, ds_info, epochs):
    if model_name in QUICKNETS.keys():
        binary_opt = tf.keras.optimizers.Adam(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=1e-2,
                # steps_per_epoch = dataset.num_examples("train") // batch_size
                warmup_steps=ds_info.steps_per_epoch * 5,
                decay_steps=ds_info.steps_per_epoch * epochs,
            )
        )
        fp_opt = tf.keras.optimizers.SGD(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=0.1,
                warmup_steps=ds_info.steps_per_epoch * 5,
                decay_steps=ds_info.steps_per_epoch * epochs,
            ),
            momentum=0.9,
        )
        optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, binary_opt),
            default_optimizer=fp_opt,
        )
    else:
        if opt_name.startswith("sgd"):
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=momentum,
                decay=weight_decay,
                nesterov="nesterov" in opt_name
            )
        elif opt_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(lr, rho=0.9, momentum=momentum, epsilon=0.0316, decay=weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD and RMSprop are supported.")
    return optimizer


def steplr(epoch, lr, gamma, step_size, init_lr):
    gamma = 0.5
    step_size = 10.0
    return init_lr * math.pow(gamma, math.floor(epoch / step_size))


def prepare_lr_scheduling(lr_scheduler, lr_gamma, lr_step_size, init_lr):
    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == "none":
        return None
    if lr_scheduler == "steplr":
        main_lr_scheduler = lambda e, lr: steplr(e, lr, lr_gamma, lr_step_size, init_lr)
    # elif lr_scheduler == "cosineannealinglr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=args.epochs - args.lr_warmup_epochs
    #     )
    # elif lr_scheduler == "exponentiallr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR supported at the moment."
        )

    # TODO also implement this!
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
    return tf.keras.callbacks.LearningRateScheduler(main_lr_scheduler)


def prepare_model(model_name, optimizer, metrics=['accuracy', 'sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'], weights=None):
    mfile = None
    try:
        model = MODELS[model_name]
    except (TypeError, KeyError) as e:
        avail = ', '.join(n for n, _ in MODELS.items())
        raise RuntimeError(f'Error when loading {model_name}! \n{e}\nAvailable models:\n{avail}')
    if weights == 'pretrained':
        print(f'Loading pretrained weights!')
        weights = 'imagenet'
    elif weights is not None and os.path.isdir(weights): # load the custom weights
        best_model = sorted([f for f in os.listdir(weights) if f.startswith('checkpoint')])[-1]
        print(f'Loading weights from {best_model}!')
        weights = os.path.join(weights, best_model)
        mfile = weights
    model = model(include_top=True, weights=weights)

    # TODO which Crossentropy to use?!
    # criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if optimizer is None: # use any default optimizer
        optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

    return model, mfile


def create_output_dir(dir, use_timestamp, config=None):
    # create log dir
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if use_timestamp:
        dir = f'{dir}_{timestamp}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # write config
    if config is not None: 
        with open(os.path.join(dir, 'config.json'), 'w') as cfg:
            config['timestamp'] = timestamp
            json.dump(config, cfg, indent=4)
    return dir


class Logger(object):
    def __init__(self, fname='logfile.txt'):
        self.terminal = sys.stdout
        self.log = open(fname, 'a')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal