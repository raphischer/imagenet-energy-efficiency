from collections import defaultdict
from multiprocessing import Process, Event
from datetime import datetime
import time
import argparse
import json

import numpy as np

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU, nvmlDeviceGetCount


def aggregate_log(fpath):
    results = defaultdict(dict)
    with open(fpath, 'r') as fc:
        log = json.load(fc)
        for gpu_id, meas in log.items():
            results[gpu_id]['start_unix'] = min(meas['timestamp']) / 1000
            results[gpu_id]['end_unix'] = max(meas['timestamp']) / 1000
            results[gpu_id]['start_utc'] = datetime.utcfromtimestamp(results[gpu_id]['start_unix']).strftime('%Y-%m-%d %H:%M:%S')
            results[gpu_id]['end_utc'] = datetime.utcfromtimestamp(results[gpu_id]['end_unix']).strftime('%Y-%m-%d %H:%M:%S')
            results[gpu_id]['duration'] = results[gpu_id]['end_unix'] - results[gpu_id]['start_unix']
            results[gpu_id]['nr_measurements'] = len(meas['timestamp'])
            for field in ['util.gpu', 'temperature', 'memory.used', 'power.draw', 'util.memory']:
                results[gpu_id][field] = {m.__name__: m(meas[field]) for m in [min, max, np.mean]}
        print(results)
    return results


def get_gpu_stats(gpu_id):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)

    # Energy
    milliWatts = nvmlDeviceGetPowerUsage(handle)
    # Memory
    memory_t = nvmlDeviceGetMemoryInfo(handle)
    # Utilization
    utilization_t = nvmlDeviceGetUtilizationRates(handle)
    tmp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    unix_time_millis = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return {
        'temperature': tmp,
        'util.gpu': utilization_t.gpu,
        'util.memory': utilization_t.memory,
        'power.draw': milliWatts / 1000.0,
        'memory.used': memory_t.used,
        'timestamp': unix_time_millis
    }


def profile(interval, logfile, stopper, gpu_id):
    nvmlInit()
    out = defaultdict(lambda: defaultdict(list))
    if gpu_id is None:
        deviceCount = nvmlDeviceGetCount()
        gpu_id = list(range(deviceCount))
    else:
        if not isinstance(gpu_id, list):
            gpu_id = [gpu_id]
    print('PROFILING FOR GPUs', gpu_id)
    i = 0
    while not stopper.is_set():
        # TODO check amount of stored data and flush out if necessary
        start = time.time()
        i += 1
        for gid in gpu_id:
            profiled = get_gpu_stats(gid)
            for key, val in profiled.items():
                out[gid][key].append(val)
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(i, 'SUCCESSFUL PROFILINGS, STORED TO', logfile)
    with open(logfile, 'w') as log:
        json.dump(out, log)


class GpuMonitoringProcess:

    def __init__(self, interval=1, outfile=None, gpu_id=None) -> None:
        self.interval = interval
        if outfile is None:
            raise NotImplementedError('Implement using a custom tmp file if no file is given!')
        self.outfile = outfile
        self.gpu_id = gpu_id

    def run(self, func):

        # TODO log manufactur information for every GPU

        stopper = Event()
        p = Process(target=profile, args=(self.interval, self.outfile, stopper, self.gpu_id))
        p.start()
        result = func()
        stopper.set() # stops loop in profiling processing
        p.join()

        # ts = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds()
        # print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

        with open(self.outfile, 'r') as tmp:
            out = json.load(tmp)

        return out, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates a given GPU profiling result")

    parser.add_argument("--log", default="/home/fischer/mnt_imagenet/models/train_2021_12_10_15_56/monitoring.json", type=str, help="dataset path")
    
    args = parser.parse_args()
    print(json.dumps(aggregate_log(args.log), indent=4))
