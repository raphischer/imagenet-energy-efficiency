from collections import defaultdict
from multiprocessing import Process, Event
from datetime import datetime
import os
import argparse
import json
import time

import numpy as np


# TODO log manufactur information for every GPU
# TODO improve CPU logs, maybe using https://github.com/djselbeck/rapl-read-ryzen


def start_monitoring(gpu_interval, cpu_interval, output_dir):
    monitoring = []
    if gpu_interval > 0:
        monitoring.append(DeviceMonitor('gpu', interval=gpu_interval, outfile=os.path.join(output_dir, 'monitoring_gpu.json')))
    if cpu_interval > 0:
        monitoring.append(DeviceMonitor('cpu', interval=cpu_interval, outfile=os.path.join(output_dir, 'monitoring_cpu.json')))
    return monitoring


def aggregate_log(fpath):
    if not os.path.isfile(fpath):
        return None
    results = defaultdict(dict)
    with open(fpath, 'r') as fc:
        log = json.load(fc)
        device_fields = [key for key in list(log.values())[0].keys() if key not in ['timestamp']]
        for gpu_id, meas in log.items():
            results[gpu_id]['start_unix'] = min(meas['timestamp']) / 1000
            results[gpu_id]['end_unix'] = max(meas['timestamp']) / 1000
            results[gpu_id]['start_utc'] = datetime.utcfromtimestamp(results[gpu_id]['start_unix']).strftime('%Y-%m-%d %H:%M:%S')
            results[gpu_id]['end_utc'] = datetime.utcfromtimestamp(results[gpu_id]['end_unix']).strftime('%Y-%m-%d %H:%M:%S')
            results[gpu_id]['duration'] = results[gpu_id]['end_unix'] - results[gpu_id]['start_unix']
            results[gpu_id]['nr_measurements'] = len(meas['timestamp'])
            for field in device_fields:
                results[gpu_id][field] = {m.__name__: m(meas[field]) for m in [min, max, np.mean, np.std]}
            if 'power_usage' in results[gpu_id]:
                results[gpu_id]['total_power_draw'] = results[gpu_id]['duration'] * results[gpu_id]['power_usage']['mean']
            else:
                results[gpu_id]['total_power_draw'] = -1
    # aggregate over all GPUs
    results['total'] = {
        'start_unix': min([val['start_unix'] for val in results.values()]),
        'end_unix': max([val['end_unix'] for val in results.values()]),
        'nr_measurements': sum([val['nr_measurements'] for val in results.values()]),
        'total_power_draw': sum([val['total_power_draw'] for val in results.values()]),
    }
    results['start_utc'] = datetime.utcfromtimestamp(results['total']['start_unix']).strftime('%Y-%m-%d %H:%M:%S')
    results['end_utc'] = datetime.utcfromtimestamp(results['total']['end_unix']).strftime('%Y-%m-%d %H:%M:%S')
    results['duration'] = results['total']['end_unix'] - results['total']['start_unix']
    return results


def monitor_gpu(interval, logfile, stopper, gpu_id):
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
    from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU, nvmlDeviceGetCount
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
            handle = nvmlDeviceGetHandleByIndex(gid)
            memory_t = nvmlDeviceGetMemoryInfo(handle)
            utilization_t = nvmlDeviceGetUtilizationRates(handle)
            out[gid]['temperature'].append(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)),
            out[gid]['util.gpu'].append(utilization_t.gpu),
            out[gid]['util.memory'].append(utilization_t.memory),
            out[gid]['power_usage'].append(nvmlDeviceGetPowerUsage(handle) / 1000.0),
            out[gid]['memory.used'].append(memory_t.used),
            out[gid]['memory.free'].append(memory_t.free),
            out[gid]['timestamp'].append((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(f'Wrote {i} GPU profilings to {logfile}')
    with open(logfile, 'w') as log:
        json.dump(out, log)


def monitor_cpu(interval, logfile, stopper, process_id):
    import psutil
    proc = psutil.Process(process_id)
    out = defaultdict(lambda: defaultdict(list))
    if not isinstance(process_id, list):
        process_id = [process_id]
    print('PROFILING FOR PROCESSES', process_id)
    i = 0
    while not stopper.is_set():
        # TODO check amount of stored data and flush out if necessary
        start = time.time()
        i += 1
        for gid in process_id:
            mem_info = proc.memory_full_info()
            out[gid]['cpu_num'].append(proc.cpu_num()),
            out[gid]['cpu_percent'].append(proc.cpu_percent()),
            for attr in ['rss', 'vms', 'shared', 'text', 'lib', 'data', 'uss', 'pss']:
                out[gid][f'mem_{attr}'].append(getattr(mem_info, attr))
            out[gid]['timestamp'].append((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(f'Wrote {i} process profilings to {logfile}')
    with open(logfile, 'w') as log:
        json.dump(out, log)


class DeviceMonitor:

    def __init__(self, device='gpu', interval=1, outfile=None, device_id=None) -> None:
        if device == 'gpu':
            prof_func = monitor_gpu
        elif device == 'cpu':
            prof_func = monitor_cpu
            device_id = os.getpid()
        else:
            raise NotADirectoryError(f'Profiling for device {device} not implemented!')
        self.interval = interval
        if outfile is None:
            raise NotImplementedError('Implement using a custom tmp file if no file is given!')
        self.outfile = outfile
        self.stopper = Event()
        self.p = Process(target=prof_func, args=(self.interval, self.outfile, self.stopper, device_id))
        self.p.start()

    def stop(self):
        self.stopper.set() # stops loop in profiling processing
        self.p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates a given GPU profiling result")

    parser.add_argument("--logs", default="/home/fischer/mnt_imagenet/models/train_2021_12_10_15_56", type=str, help="directory with logs")
    
    args = parser.parse_args()
    print(json.dumps(aggregate_log(os.path.join(args.logs, 'monitoring_cpu.json')), indent=4))
    print(json.dumps(aggregate_log(os.path.join(args.logs, 'monitoring_gpu.json')), indent=4))
