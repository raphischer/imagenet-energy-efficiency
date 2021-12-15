from datetime import datetime
import json
import os
import random as python_random
import sys

import tensorflow as tf
import numpy as np


def fix_seed(seed):
    if seed == -1:
        seed = python_random.randint(0, 2**32 - 1)

    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    return seed


def prepare_model(model_name, opt_name, lr, momentum, weight_decay, weights=None):
    try:
        model = tf.keras.applications.__dict__[model_name]
    except (TypeError, KeyError) as e:
        avail = ', '.join(n for n, e in tf.keras.applications.__dict__.items() if callable(e))
        raise RuntimeError(f'Error when loading {model_name}! \n{e}\nAvailable models:\n{avail}')
    if weights is not None and os.path.isdir(weights): # load the custom weights
        weights = os.path.join(weights, 'checkpoint.hdf5')
    model = model(include_top=True, weights=weights)

    # TODO which Crossentropy to use?!
    # criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

    return model


def create_output_dir(dir, use_timestamp, config=None):
    # create log dir
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
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