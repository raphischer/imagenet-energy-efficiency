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


def aggregate_results(directory, train_directories=None):
    res = {'directory': directory}
    try:
        with open(os.path.join(directory, 'config.json'), 'r') as cf:
            res['config'] = json.load(cf)
        if os.path.basename(directory).startswith('eval'):
            for split in ['train', 'validation']:
                res[split] = {}
                res[split]['monitoring_gpu'] = aggregate_log(os.path.join(directory, f'{split}_monitoring_gpu.json'))
                res[split]['monitoring_cpu'] = aggregate_log(os.path.join(directory, f'{split}_monitoring_cpu.json'))
                res[split]['monitoring_rapl'] = aggregate_log(os.path.join(directory, f'{split}_monitoring_rapl.json'))
                res[split]['results'] = read_metrics(os.path.join(directory, f'{split}_results.json'))
                if res[split]['monitoring_gpu'] is None or len(res[split]['results']) == 0: # monitoring not yet finished or errors
                    res[split]['duration'] = (datetime.now() - datetime.strptime(res["config"]["timestamp"], "%Y_%m_%d_%H_%M_%S")).total_seconds()
                else:
                    res[split]['duration'] = res[split]['results']['end'] - res[split]['results']['start']
            if os.path.basename(res['config']['eval_model']) == '': # reformat paths ending with / behind the directory TODO remove later
                res['config']['eval_model'] = os.path.dirname(res['config']['eval_model'])
            train_dir = os.path.join(train_directories, os.path.basename(res['config']['eval_model']))
            if os.path.isdir(train_dir):
                res['training'] = aggregate_results(train_dir)
        else:
            res['monitoring_gpu'] = aggregate_log(os.path.join(directory, 'monitoring_gpu.json'))
            res['monitoring_cpu'] = aggregate_log(os.path.join(directory, 'monitoring_cpu.json'))
            res['monitoring_rapl'] = aggregate_log(os.path.join(directory, 'monitoring_rapl.json'))
            res['results'] = read_metrics(os.path.join(directory, 'results.json'))
            if res['monitoring_gpu'] is None or len(res['results']) == 0: # monitoring not yet finished or errors
                res['duration'] = (datetime.now() - datetime.strptime(res["config"]["timestamp"], "%Y_%m_%d_%H_%M_%S")).total_seconds()
            else:
                res['duration'] = res['results']['end'] - res['results']['start']
    except Exception as e:
        res['Error'] = str(e)
    return res


def aggregate_all_results(directory, train_directories=None):
    results = {}
    for dir in sorted(os.listdir(directory)):
        results[dir] = aggregate_results(os.path.join(directory, dir), train_directories)
    return results


def print_train_results(results):
    if len(results) < 1:
        return
    print('\n\nTRAINING\n\n          Directory       -               Model Info                -        Configuration       -     Duration    -   Acc  - Power Draw')
    for dir, res in results.items():
        # print different output depending on accessing train or eval directory
        if 'Error' in res:
            print(f'{dir} - ERROR - {res["Error"]}')
        else:
            if len(res["results"]) > 0:
                checkpoints = sorted([cp for cp in os.listdir(res['directory']) if 'checkpoint' in cp])
                epochs = f'{int(checkpoints[-1].split("_")[1]):>4}'
                gpu_draw = f'GPU {res["monitoring_gpu"]["total"]["total_power_draw"] / 3600000:4.1f} kWh'
                model_info = f'{res["config"]["model"]:<16} {res["results"]["model"]["fsize"] * 1e-6:5.1f} MB {res["results"]["model"]["params"] * 1e-6:5.1f}M params'
                acc = f'{res["results"]["history"]["accuracy"][-1]*100:5.1f}%'
            else:
                epochs = f'n.a.'
                gpu_draw, acc = ' n.a. ', '  n.a.'
                model_info = f'{res["config"]["model"]:<16} n.a.     n.a.'
            cfg = f'{epochs} epochs, {res["config"]["preprocessing"]:<8} prepr'
            duration = str(timedelta(seconds=res["duration"]))[:-10]
            print(f'{dir} - {model_info:<39} - {cfg} - {duration:>14}h - {acc} - {gpu_draw}')


def print_eval_results(results):
    if len(results) < 1:
        return
    print('\n\nEVALUATION\n\n          Directory       -    Model Info    -       Configuration       -          Training Info          -    Evaluation Train Data  - Evaluation Validation Data')
    for dir, values in results.items():
        if 'Error' in values:
            print(f'{dir} - ERROR - {values["Error"]}')
        else:
            substrings = [f'{dir:<25}', f'{values["config"]["model"]:<16}']
            if 'training' not in values:
                substrings.append('        pretrained       ')
                substrings.append('                n.a.           ')
            else:
                res = values['training']
                substrings.append(f'{res["config"]["epochs"]:<3} epochs, {res["config"]["preprocessing"]:<7} prepr')
                duration = str(timedelta(seconds=res["duration"]))[:-10]
                acc = f'{res["results"]["history"]["accuracy"][-1]*100:5.1f}%'
                gpu_draw = f'{res["monitoring_gpu"]["total"]["total_power_draw"] / 3600000:4.1f} kWh'
                substrings.append(f'{duration:>13}h {acc} {gpu_draw}')
            # access evaluation results
            for split in ['train', 'validation']:
                if split in values:
                    res = values[split]
                    if "metrics" in res["results"] and res["results"]["metrics"] is not None:
                        gpu_draw = f'{res["monitoring_gpu"]["total"]["total_power_draw"] / 3600:5.1f} Wh'
                        acc = f'{res["results"]["metrics"]["accuracy"]*100:<4.3f}% acc'
                    else:
                        gpu_draw = '  n.a.  '
                        acc = f'{"n.a.":<9}'

                    substrings.append(f'{str(timedelta(seconds=res["duration"]))[2:-7]}m {acc} {gpu_draw}')
                else:
                    substrings.append('           n.a.         ')
            print(" - ".join(substrings))


def main(directory, train_directories, output):
    results = aggregate_all_results(directory, train_directories)
    if len(results) > 0:
        with open(output, 'w') as rf:
            json.dump(results, rf, indent=4)
        print_train_results({key: res for key, res in results.items() if key.startswith('train')})
        print_eval_results({key: res for key, res in results.items() if key.startswith('eval')})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="/raid/fischer/eval", type=str, help="directory with experiments")
    parser.add_argument("--train-dirs", default="/raid/fischer/train50", type=str, help="directory with original training experiments, only used for eval results")
    parser.add_argument("--output", default="results.json", type=str, help="form of output, either directory for generating plots, or empty string for command line")

    args = parser.parse_args()
    main(args.directory, args.train_dirs, args.output)
