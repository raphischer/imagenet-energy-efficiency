from datetime import datetime
import json
import os
import random as python_random
import sys

import tensorflow as tf
import numpy as np


class TimestampOnEpochEnd(tf.keras.callbacks.Callback):

    def __init__(self, path):
        super(TimestampOnEpochEnd, self).__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        timestamp = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
        if 'timestamp' not in self.model.history.history:
            self.model.history.history['timestamp'] = []
        self.model.history.history['timestamp'].append(timestamp)


class PatchedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def fix_seed(seed):
    if seed == -1:
        seed = python_random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    return seed


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