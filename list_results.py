import os
import argparse
import json
from datetime import datetime, timedelta

from create_plots import create_plots
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
        # TODO move this to train (write designated results file)
        from util import prepare_model
        if os.path.basename(directory).startswith('train'):
            model = prepare_model(cfg['model'], None, weights=directory)
            cfg['model_params'] = model.count_params()
            cfg['model_fsize'] = max([os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory) if f.startswith('check')])
        return cfg
    # except Exception:
    #     return None


def aggregate_all_results(directory, train_directories, after=None):
    train = {}
    evals = {}
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
            if dir.startswith('train'):
                train[dir] = res
            elif dir.startswith('eval'):
                train_dir = os.path.basename(res["model_dir"])
                if train_dir not in evals:
                    # load the training results from the stored model path
                    train_res = os.path.join(train_directories, train_dir)
                    evals[train_dir] = {'training': aggregate_results(train_res)}
                assert res["split"] not in evals[train_dir], f'Error! Already aggregated {res["split"]} results for {train_dir}'
                evals[train_dir][res["split"]] = res
    return train, evals    
    

def list_results(directory, train_directories, after=None, plots=''):
    if os.path.isfile('results_train.json'):
        with open('results_train.json', 'r') as res:
            train = json.load(res)
    if os.path.isfile('results_eval.json'):
        with open('results_eval.json', 'r') as res:
            evals = json.load(res)
    else:
        train, evals = aggregate_all_results(directory, train_directories, after)
        if len(train) > 0:
            with open('results_train.json', 'w') as res:
                json.dump(train, res, indent=4)
        if len(evals) > 0:
            with open('results_eval.json', 'w') as res:
                json.dump(evals, res, indent=4)
    # create plots
    if len(plots) > 0:
        if not os.path.isdir(plots):
            os.makedirs(plots)
        create_plots(evals, plots)
    print('\n\nTRAINING\n\n          Directory       -               Model Info                -       Configuration       -     Duration    -  Accuracy  - Power Draw')
    # parse only training directories
    for dir, res in train.items():
        prep = 'simple' if res['use_simple'] else 'complex'
        # print different output depending on accessing train or eval directory
        gpu_draw = 'n.a.'
        if res['gpu_monitoring'] is not None:
            gpu_id, gpu_res = list(res["gpu_monitoring"].items())[0]
            gpu_draw = f'GPU {gpu_id} {gpu_res["total_power_draw"] / 3600000:3.1f} kWh'
        model_info = f'{res["model"]:<16} {res["model_fsize"] * 1e-6:5.1f} MB {res["model_params"] * 1e-6:5.1f}M params'
        cfg = f'{res["epochs"]:<3} epochs, {prep:<7} prepr'
        acc = f'acc {res["train_metrics"]["accuracy"]:<6}'
        print(f'{dir} - {model_info} - {cfg} - {res["duration"][:-10]:>14}h - {acc} - {gpu_draw}')
    
    # parse evaluation directories
    print('\n\nEVALUATION\n\n          Directory       -    Model Info    -       Configuration       -            Training Info            -   Evaluation Train Data  - Evaluation Validation Data')
    for dir, values in evals.items():
        substrings = [dir]
        # access training results
        res = values['training']
        substrings.append(f'{res["model"]:<16}')
        substrings.append(f'{res["epochs"]:<3} epochs, {"simple" if res["use_simple"] else "complex":<7} prepr') # preprocessing args
        acc = f'{res["train_metrics"]["accuracy"]*100:<5.1f}% acc'
        gpu_draw = f'{list(res["gpu_monitoring"].values())[0]["total_power_draw"] / 3600000:3.1f} kWh'
        substrings.append(f'{res["duration"][:-10]:>14}h {acc} {gpu_draw}')
        # access evaluation results
        for split in ['train', 'validation']:
            res = values[split]
            if res['gpu_monitoring'] is not None:
                gpu_id, gpu_res = list(res["gpu_monitoring"].items())[0]
                gpu_draw = f'{gpu_res["total_power_draw"] / 3600:3.1f} Wh'
            if res["eval_metrics"] is not None:
                acc = f'{res["eval_metrics"]["accuracy"]*100:<4.1f}% acc'
            else:
                acc = f'{"n.a."} acc'
            substrings.append(f'{res["duration"][2:-7]}m {acc} {gpu_draw}')
        print(" - ".join(substrings))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="/raid/fischer/eval", type=str, help="directory with experiments")
    parser.add_argument("--train-dirs", default="/raid/fischer/dnns", type=str, help="directory with training experiments, only used for eval results")
    parser.add_argument("--output", default="plots", type=str, help="form of output, either directory for generating plots, or empty string for command line")
    parser.add_argument("--after", default="2021_12_20", type=str, help="earliest timestamp to list")

    args = parser.parse_args()
    list_results(args.directory, args.train_dirs, args.after, args.output)
