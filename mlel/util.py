from datetime import datetime
import json
import os
import random as python_random
import sys
import pkg_resources

import numpy as np


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
    return seed


def create_output_dir(dir, config=None):
    # create log dir
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir = f'{dir}_{timestamp}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # write config
    if config is not None: 
        with open(os.path.join(dir, 'config.json'), 'w') as cfg:
            config['timestamp'] = timestamp
            json.dump(config, cfg, indent=4)
    # write installed packages
    with open(os.path.join(dir, 'requirements.txt'), 'w') as req:
        # req.write(f'# {sys.version}\n') TODO rework this
        for pkg in pkg_resources.working_set:
            req.write(f'{pkg.key}=={pkg.version}\n')
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