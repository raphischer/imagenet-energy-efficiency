import os
import argparse
import json
from datetime import datetime, timedelta
import shutil
import tarfile

from mlel.monitoring import aggregate_log
from mlel.util import basename


def read_metrics(filepath, default_path=None):
    if not os.path.isfile(filepath):
        if default_path is not None:
            shutil.copyfile(os.path.join(default_path, basename(filepath)), filepath)
        else:
            return {}
    with open(filepath, 'r') as logf:
        return json.load(logf)


def read_requirements(filepath, default_path=None):
    if not os.path.isfile(filepath):
        if default_path is not None:
            shutil.copyfile(os.path.join(default_path, basename(filepath)), filepath)
        else:
            return {}
    with open(filepath, 'r') as reqf:
        return [line.strip() for line in reqf.readlines()]


def aggregate_results(directory, train_directories=None):
    res = {'directory_name': basename(directory)}
    try:
        with open(os.path.join(directory, 'config.json'), 'r') as cf:
            res['config'] = json.load(cf)
            res['execution_platform'] = read_metrics(os.path.join(directory, 'execution_platform.json'))
            res['requirements'] = read_requirements(os.path.join(directory, 'requirements.txt'))
        if basename(directory).startswith('infer'):
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
            train_dir = os.path.join(train_directories, basename(res['config']['infer_model']))
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


def process_directory(directory, train_directories=None, output_log_dir=None, output_agglog_dir=None):
    # create summary
    print('Processing', directory)
    if output_agglog_dir is not None and len(output_agglog_dir) > 0 and os.path.isdir(output_agglog_dir):
        agglog_name = os.path.join(output_agglog_dir, basename(directory) + '.json')
        if os.path.isfile(agglog_name):
            print('WARNING!', agglog_name, 'already exists, so will not create new!')
            with open(agglog_name, 'r') as agglog:
                res = json.load(agglog)
        else:
            res = aggregate_results(directory, train_directories)
            if output_log_dir is not None and len(output_log_dir) > 0 and os.path.isdir(output_log_dir):
                res['log_directory'] = os.path.join(output_log_dir, basename(directory) + '.tar.gz')
            with open(agglog_name, 'w') as agglog:
                json.dump(res, agglog, indent=4)
    else:
        res = aggregate_results(directory, train_directories)
    # create tar
    if output_log_dir is not None and os.path.isdir(output_log_dir):
        log_tar_name = os.path.join(output_log_dir, basename(directory) + '.tar.gz')
        if not os.path.exists(log_tar_name):
            with tarfile.open(log_tar_name, 'w:gz') as tar:
                for fname in os.listdir(directory):
                    tar.add(os.path.join(directory, fname))
        else:
            print('WARNING!', log_tar_name, 'already exists, so will not create new!')
    return res


def process_all_subdirectories(directory, train_directories=None, output_log_dir=None, output_agglog_dir=None):
    results = {}
    for dir in sorted(os.listdir(directory)):
        results[dir] = process_directory(os.path.join(directory, dir), train_directories, output_log_dir, output_agglog_dir)
    return results


def print_train_results(results):
    if len(results) < 1:
        return
    print('\n\nTRAINING\n\n          Directory       -               Model Info                -        Configuration       -     Duration    -   Acc  - Power Draw')
    for dir, res in results.items():
        # print different output depending on accessing train or infer directory
        if 'Error' in res:
            print(f'{dir} - ERROR - {res["Error"]}')
        else:
            if len(res["results"]) > 0:
                epochs = f'{len(res["results"]["history"]["loss"]):>4}'
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
        # print(dir, f"{res['config']['model']:<16}", ' '.join([f"{no:2.1f}" for no in res["results"]["history"]["loss"][:50]]))


def print_inference_results(results):
    if len(results) < 1:
        return
    print('\n\nINFERENCE\n\n          Directory       -    Model Info    -       Configuration       -          Training Info          -    Inference Train Data  - Inference Validation Data')
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
            # access inference results
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


def main(directory, train_directories, output_log_dir=None, output_agglog_dir=None, clean=False):
    if clean:
        for rootdir in [output_log_dir, output_agglog_dir]:
            if os.path.isdir(rootdir):
                for subdir in os.listdir(rootdir):
                    if os.path.isfile(os.path.join(rootdir, subdir)):
                        os.remove(os.path.join(rootdir, subdir))
                    else:
                        shutil.rmtree(os.path.join(rootdir, subdir))
    results = process_all_subdirectories(directory, train_directories, output_log_dir, output_agglog_dir)
    if len(results) > 0:
        print_train_results({key: res for key, res in results.items() if key.startswith('train')})
        print_inference_results({key: res for key, res in results.items() if key.startswith('infer')})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="/raid/fischer/eval", type=str, help="directory with experiments")
    parser.add_argument("--train-dirs", default="/raid/fischer/train50", type=str, help="directory with original training experiments, can be used for the inference results")
    parser.add_argument("--output-log-dir", default="", type=str, help="directory where the logs shall be stored (.tar.gz archives)")
    parser.add_argument("--output-agglog-dir", default="results", type=str, help="directory where experiments log aggregates (json format) are created")
    parser.add_argument("--clean", action='store_true', help="set to first delete all content in given output directories")

    args = parser.parse_args()
    main(args.directory, args.train_dirs, args.output_log_dir, args.output_agglog_dir, args.clean)
