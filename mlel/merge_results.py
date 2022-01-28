import os
import argparse
import json
from datetime import datetime, timedelta

from mlel.monitoring import aggregate_log


def read_metrics(filepath):
    if not os.path.isfile(filepath):
        return {}
    with open(filepath, 'r') as logf:
        return json.load(logf)


def aggregate_results(directory):
    with open(os.path.join(directory, 'config.json'), 'r') as cf:
        cfg = json.load(cf)
    cfg['gpu_monitoring'] = aggregate_log(os.path.join(directory, 'monitoring_gpu.json'))
    cfg['cpu_monitoring'] = aggregate_log(os.path.join(directory, 'monitoring_cpu.json'))
    cfg['results'] = read_metrics(os.path.join(directory, 'results.json'))
    if cfg['gpu_monitoring'] is None or len(cfg['results']) == 0: # training not yet finished or errors
        cfg['duration'] = (datetime.now() - datetime.strptime(cfg["timestamp"], "%Y_%m_%d_%H_%M_%S")).total_seconds()
    else:
        cfg['duration'] = cfg['results']['end'] - cfg['results']['start']
    cfg['directory'] = directory
    return cfg


def aggregate_all_results(directory, train_directories):
    train = {}
    evals = {}
    for dir in sorted(os.listdir(directory)):
        res = aggregate_results(os.path.join(directory, dir))
        if res is not None:
            if dir.startswith('train'):
                train[dir] = res
            elif dir.startswith('eval'):
                if res["eval_model"][-1] == '/':
                    eval_model = os.path.basename(os.path.dirname(res["eval_model"]))
                else:
                    eval_model = os.path.basename(res["eval_model"])
                train_dir = os.path.join(train_directories, eval_model)
                if eval_model not in evals:
                    evals[eval_model] = {}
                if os.path.isdir(train_dir):
                    # load the training results from the stored model path
                    evals[eval_model]['training'] = aggregate_results(train_dir)
                if res["split"] in evals[eval_model]:
                    print(f'Warning! Already found {res["split"]} results for {eval_model} in {evals[eval_model][res["split"]]["directory"]}, so skipping {dir}')
                else:
                    evals[eval_model][res["split"]] = res
    return train, evals


def print_train_results(results):
    print('\n\nTRAINING\n\n          Directory       -               Model Info                -        Configuration       -     Duration    -   Acc  - Power Draw')
    for dir, res in results.items():
        # print different output depending on accessing train or eval directory
        gpu_draw = 'n.a.'
        if len(res["results"]) > 0:
            checkpoints = sorted([cp for cp in os.listdir(res['directory']) if 'checkpoint' in cp])
            epochs = f'{int(checkpoints[-1].split("_")[1]):>4}'
            gpu_draw = f'GPU {res["gpu_monitoring"]["total"]["total_power_draw"] / 3600000:4.1f} kWh'
            model_info = f'{res["model"]:<16} {res["results"]["model"]["fsize"] * 1e-6:5.1f} MB {res["results"]["model"]["params"] * 1e-6:5.1f}M params'
            acc = f'{res["results"]["history"]["accuracy"][-1]*100:5.1f}%'
        else:
            epochs = f'n.a.'
            gpu_draw, acc = ' n.a. ', '  n.a.'
            model_info = f'{res["model"]:<16} n.a.     n.a.'
        cfg = f'{epochs} epochs, {res["preprocessing"]:<8} prepr'
        duration = str(timedelta(seconds=res["duration"]))[:-10]
        print(f'{dir} - {model_info:<39} - {cfg} - {duration:>14}h - {acc} - {gpu_draw}')


def print_eval_results(results):
    print('\n\nEVALUATION\n\n          Directory       -    Model Info    -       Configuration       -          Training Info          -    Evaluation Train Data  - Evaluation Validation Data')
    for dir, values in results.items():
        substrings = [f'{dir:<25}']
        if 'training' not in values:
            substrings.append('                    pretrained              ')
            substrings.append('                n.a.           ')
        else:
            substrings.append(f'{values["train"]["model"]:<16}')
            res = values['training']
            substrings.append(f'{res["epochs"]:<3} epochs, {res["preprocessing"]:<7} prepr')
            duration = str(timedelta(seconds=res["duration"]))[:-10]
            acc = f'{res["results"]["history"]["accuracy"][-1]*100:5.1f}%'
            gpu_draw = f'{res["gpu_monitoring"]["total"]["total_power_draw"] / 3600000:4.1f} kWh'
            substrings.append(f'{duration:>13}h {acc} {gpu_draw}')
        # access evaluation results
        for split in ['train', 'validation']:
            if split in values:
                res = values[split]
                if "metrics" in res["results"] and res["results"]["metrics"] is not None:
                    gpu_draw = f'{res["gpu_monitoring"]["total"]["total_power_draw"] / 3600:5.1f} Wh'
                    acc = f'{res["results"]["metrics"]["accuracy"]*100:<4.1f}% acc'
                else:
                    gpu_draw = '  n.a.  '
                    acc = f'{"n.a.":<9}'

                substrings.append(f'{str(timedelta(seconds=res["duration"]))[2:-7]}m {acc} {gpu_draw}')
            else:
                substrings.append('           n.a.         ')
        print(" - ".join(substrings))


def main(directory, train_directories, output=''):
    if len(output) > 0 and not os.path.isdir(output):
        os.makedirs(output)
    train, evals = aggregate_all_results(directory, train_directories)
    if len(train) > 0:
        if len(output) > 0:
            with open(os.path.join(output, 'results_train.json'), 'w') as rf:
                json.dump(train, rf, indent=4)
        print_train_results(train)
    if len(evals) > 0:
        if len(output) > 0:
            with open(os.path.join(output, 'results_eval.json'), 'w') as rf:
                json.dump(evals, rf, indent=4)
        print_eval_results(evals)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="/raid/fischer/eval", type=str, help="directory with experiments")
    parser.add_argument("--train-dirs", default="/raid/fischer/dnns", type=str, help="directory with original training experiments, only used for eval results")
    parser.add_argument("--output", default="", type=str, help="form of output, either directory for generating plots, or empty string for command line")

    args = parser.parse_args()
    main(args.directory, args.train_dirs, args.output)
