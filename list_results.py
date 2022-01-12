import os
import argparse
import json
from datetime import datetime, timedelta

from monitoring import aggregate_log


def read_metrics_from_train_log(filepath):
    with open(filepath, 'r') as logf:
        last_epoch_line = logf.readlines()[-2]
        metr_str = last_epoch_line.split(' - ', 3)[3][:-1].split(' - ')
        metrics = {metr.split(': ')[0]: float(metr.split(': ')[1]) for metr in metr_str}
        return metrics


def read_metrics(filepath):
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r') as logf:
        return json.load(logf)


def aggregate_results(directory):
    # try:
        with open(os.path.join(directory, 'config.json'), 'r') as cf:
            cfg = json.load(cf)
        cfg['gpu_monitoring'] = aggregate_log(os.path.join(directory, 'monitoring_gpu.json'))
        cfg['cpu_monitoring'] = aggregate_log(os.path.join(directory, 'monitoring_cpu.json'))
        if cfg['gpu_monitoring'] is None: # training not yet finished
            # TODO this does not read training, but logfile metrics!
            cfg['train_metrics'] = {'accuracy': 'n.a.'}
            cfg['duration'] = str(datetime.now() - datetime.strptime(cfg["timestamp"], "%Y_%m_%d_%H_%M_%S"))
        else:
            cfg['train_metrics'] = read_metrics_from_train_log(os.path.join(directory, 'logfile.txt'))
            cfg['duration'] = str(timedelta(seconds=list(cfg['gpu_monitoring'].values())[0]["duration"]))
        cfg['eval_metrics'] = read_metrics(os.path.join(directory, 'results.json'))
        return cfg
    # except Exception:
    #     return None
    

def list_results(directory, after=None):
    for dir in sorted(os.listdir(directory)):
        if after is not None:
            with open(os.path.join(directory, dir, 'config.json'), 'r') as cf:
                cfg = json.load(cf)
            min_time = datetime.strptime(after, '%Y_%m_%d')
            exp_time = datetime.strptime(cfg['timestamp'], '%Y_%m_%d_%H_%M_%S')
            if exp_time < min_time:
                continue
        res = aggregate_results(os.path.join(directory, dir))
        if res is not None:
            prep = 'simple' if res['use_simple'] else 'complex'
            # print different output depending on accessing train or eval directory
            gpu_draw = 'n.a.'
            if dir.startswith('train'):
                if res['gpu_monitoring'] is not None:
                    gpu_id, gpu_res = list(res["gpu_monitoring"].items())[0]
                    gpu_draw = f'GPU {gpu_id} {gpu_res["total_power_draw"] / 3600000:3.1f} kWh'
                print(f'{dir} - {res["model"]:<16} - {res["epochs"]:<3} epochs, {prep:<7} prepr - {res["duration"][:-10]:>14}h - acc {res["train_metrics"]["accuracy"]:<6} - {gpu_draw}')
        
            elif dir.startswith('eval'):
                if res['gpu_monitoring'] is not None:
                    gpu_id, gpu_res = list(res["gpu_monitoring"].items())[0]
                    gpu_draw = f'GPU {gpu_id} {gpu_res["total_power_draw"] / 3600:3.1f} Wh'
                if res["eval_metrics"] is not None:
                    acc = f'{res["split"][:5]} acc {res["eval_metrics"]["accuracy"]:<6.3f}'
                else:
                    acc = f'{res["split"][:5]} acc {"n.a.":<6}'
                print(f'{dir} - {res["model"]:<16} - {res["epochs"]:<3} epochs, {prep:<7} prepr - {res["duration"][:-10]:>6}h - {acc} - {gpu_draw}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="/raid/fischer/eval", type=str, help="directory with experiments")
    parser.add_argument("--after", default="2021_12_20", type=str, help="earliest timestamp to list")

    args = parser.parse_args()
    list_results(args.directory, args.after)
