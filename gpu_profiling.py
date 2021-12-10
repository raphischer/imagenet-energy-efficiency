from collections import defaultdict
from multiprocessing import Queue, Process
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


class GpuMonitoringProcess:

    def __init__(self, interval=1, timeout=365 * 24 * 60 * 60 * 10) -> None:
        self.interval = interval
        self.timeout = timeout # 10 years

    def run(self, func):
        queue = Queue()

        p = MonitoringProcess(lambda: get_gpu_stats_nvml_py(), queue, self.interval, self.timeout)
        p.start()
        result = func()
        p.terminate()

        out = defaultdict(lambda: defaultdict(list))
        while not queue.empty():
            x = queue.get()
            for gpu_id, values in x.items():
                for key, val in values.items():
                    out[gpu_id][key].append(val)

        return out, result


class MonitoringProcess(Process):
    def __init__(self, measurement_fn, queue, interval, timeout):
        """
        This allows to monitor measurements via
        :param measurement_fn:
        :param queue:
        :param interval:
        :param timeout: seconds to run the process (Default is 10 years:D)
        """
        super(MonitoringProcess, self).__init__()

        self.measurement_fn = measurement_fn
        self.queue = queue
        self.interval = interval

        self.timeout = timeout

    def run(self):

        nvmlInit()

        process_start = time.time()

        while time.time() < process_start + self.timeout:
            start = time.time()

            measurement = self.measurement_fn()
            self.queue.put(measurement)

            measure_duration = time.time() - start
            sleep_time = self.interval - measure_duration

            if sleep_time > 0:
                time.sleep(sleep_time)


def get_gpu_stats_nvml_py(gpu_id=0):
    results = {}
    deviceCount = nvmlDeviceGetCount()

    for gpu_id in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(gpu_id)

        # Energy
        milliWatts = nvmlDeviceGetPowerUsage(handle)

        # Memory
        memory_t = nvmlDeviceGetMemoryInfo(handle)
        # Utilization
        utilization_t = nvmlDeviceGetUtilizationRates(handle)

        tmp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

        unix_time_millis = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0

        results[gpu_id] =  {
            'temperature': tmp,
            'util.gpu': utilization_t.gpu,
            'util.memory': utilization_t.memory,
            'power.draw': milliWatts / 1000.0,
            'memory.used': memory_t.used,
            'timestamp': unix_time_millis
        }
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates a given GPU profiling result")

    parser.add_argument("--logpath", default="/home/fischer/mnt_imagenet/models/train_2021_12_10_15_56/monitoring.json", type=str, help="dataset path")
    
    args = parser.parse_args()
    print(aggregate_log(args.logpath))