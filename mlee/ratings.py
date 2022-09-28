import os
import json
from statistics import mean
from matplotlib.pyplot import axis

import numpy as np


HIGHER_BETTER = [
    'top1_val',
    'top5_val',
]
HARDWARE_NAMES = {
    'NVIDIA A100-SXM4-40GB': 'A100',
    'Quadro RTX 5000': 'RTX 5000',
    'Intel(R) Xeon(R) W-2155 CPU @ 3.30GHz': 'Xeon(R) W-2155',
    'AMD EPYC 7742 64-Core Processor': 'EPYC 7742'
}
MODEL_INFO = {
    'ResNet50':          {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet101':         {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet152':         {'epochs': 90, 'url': 'https://arxiv.org/abs/1512.03385'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG16':             {'epochs': 90, 'url': 'https://arxiv.org/abs/1409.1556'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG19':             {'epochs': 90, 'url': 'https://arxiv.org/abs/1409.1556'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'EfficientNetB0':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB1':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB2':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB3':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB4':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB5':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB6':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'EfficientNetB7':    {'epochs': None, 'url': 'https://arxiv.org/pdf/1905.11946.pdf'}, # no information on epochs
    'RegNetX400MF':      {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX32GF':       {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX8GF':        {'epochs': 100, 'url': 'https://arxiv.org/abs/2003.13678'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext50':         {'epochs': 100, 'url': 'https://arxiv.org/abs/1611.05431'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext101':        {'epochs': 100, 'url': 'https://arxiv.org/abs/1611.05431'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'DenseNet121':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'DenseNet169':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'DenseNet201':       {'epochs': 90, 'url': 'https://arxiv.org/pdf/1608.06993'},
    'Xception':          {'epochs': None, 'url': 'https://arxiv.org/abs/1610.02357'}, # no information on epochs
    'InceptionResNetV2': {'epochs': 200, 'url': 'https://arxiv.org/abs/1602.07261'},
    'InceptionV3':       {'epochs': 100, 'url': 'https://arxiv.org/abs/1512.00567'},
    'NASNetMobile':      {'epochs': 100, 'url': 'https://arxiv.org/pdf/1707.07012'},
    'MobileNetV2':       {'epochs': 300, 'url': 'https://arxiv.org/abs/1801.04381'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Small':  {'epochs': 600, 'url': 'https://arxiv.org/pdf/1905.02244'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Large':  {'epochs': 600, 'url': 'https://arxiv.org/pdf/1905.02244'}, # https://github.com/pytorch/vision/tree/main/references/classification
    'QuickNetSmall':     {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'}, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNet':          {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'}, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNetLarge':     {'epochs': 600, 'url': 'https://arxiv.org/abs/2011.09398'} # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
}
TASK_TYPES = {
    'infer': 'inference',
    'train': 'training',
}
TASK_METRICS_CALCULATION = {        # boolean informs whether given task log is used (True), or if the corresponding inference log shall be used instead
    'inference': {
        'parameters':               (True,  lambda model_log: calc_parameters(model_log)),
        'gflops':                   (True,  lambda model_log: calc_gflops(model_log)),
        'fsize':                    (True,  lambda model_log: calc_fsize(model_log)),
        'inference_power_draw':     (True,  lambda model_log: calc_power_draw(model_log)),
        'inference_time':           (True,  lambda model_log: calc_inf_time(model_log)),
        'top1_val':                 (True,  lambda model_log: calc_accuracy(model_log)),
        'top5_val':                 (True,  lambda model_log: calc_accuracy(model_log, top5=True))
    },
    'training': {
        'parameters':               (True,  lambda model_log: calc_parameters(model_log)),
        'gflops':                   (True,  lambda model_log: calc_gflops(model_log)),
        'fsize':                    (True,  lambda model_log: calc_fsize(model_log)),
        'train_power_draw_epoch':   (True,  lambda model_log: calc_power_draw_train(model_log, True)),
        'train_power_draw':         (True,  lambda model_log: calc_power_draw_train(model_log)),
        'train_time_epoch':         (True,  lambda model_log: calc_time_train(model_log, True)),
        'train_time':               (True,  lambda model_log: calc_time_train(model_log)),
        'top1_val':                 (False, lambda model_val_log: calc_accuracy(model_val_log)),
        'top5_val':                 (False, lambda model_val_log: calc_accuracy(model_val_log, top5=True))
    }
}


def load_backend_info(backend): # TODO access from train / infer scripts, and log info during experiment
    info_fname = os.path.join(os.path.dirname(__file__), f'ml_{backend}', 'info.json')
    with open(info_fname, 'r') as info_f:
        return json.load(info_f)


def get_environment_key(log):
    backend = load_backend_info(log['config']['backend'])
    backend_version = 'n.a.'
    for package in backend["Packages"]:
        for req in log['requirements']:
            if req.split('==')[0].replace('-', '_') == package.replace('-', '_'):
                backend_version = req.split('==')[1]
                break
        else:
            continue
        break
    n_gpus = len(log['execution_platform']['GPU'])
    if len(log['execution_platform']['GPU']) > 0:
        gpu_name = HARDWARE_NAMES[log['execution_platform']['GPU']['0']['Name']]
        name = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    else:
        name = HARDWARE_NAMES[log['execution_platform']['Processor']]
    return f'{name} - {backend["Name"]} {backend_version}'


def calculate_compound_rating(ratings, mode, meanings=None):
    if isinstance(ratings, dict): # model summary given instead of list of ratings
        weights = [val['weight'] for val in ratings.values() if isinstance(val, dict) and 'rating' in val if val['weight'] > 0]
        weights = [w / sum(weights) for w in weights]
        ratings = [val['rating'] for val in ratings.values() if isinstance(val, dict) and 'rating' in val if val['weight'] > 0]
    else:
        weights = [1.0 / len(ratings) for _ in ratings]
    if meanings is None:
        meanings = np.arange(np.max(ratings) + 1, dtype=int)
    round_m = np.ceil if 'pessimistic' in mode else np.floor # optimistic
    if mode == 'best':
        return meanings[min(ratings)] # TODO no weighting here
    if mode == 'worst':
        return meanings[max(ratings)] # TODO no weighting here
    if 'median' in mode:
        asort = np.argsort(ratings)
        weights = np.array(weights)[asort]
        ratings = np.array(ratings)[asort]
        cumw = np.cumsum(weights)
        for i, (cw, r) in enumerate(zip(cumw, ratings)):
            if cw == 0.5:
                return meanings[int(round_m(np.average([r, ratings[i + 1]])))]
            if cw > 0.5 or (cw < 0.5 and cumw[i + 1] > 0.5):
                return meanings[r]
    if 'mean' in mode:
        return meanings[int(round_m(np.average(ratings, weights=weights)))]
    if mode == 'majority':
        return meanings[np.argmax(np.bincount(ratings))]
    raise NotImplementedError('Rating Mode not implemented!', mode)


def value_to_index(value, ref, metric_key):
    #      i = v / r                     OR                i = r / v
    try:
        return value / ref if metric_key in HIGHER_BETTER else ref / value
    except:
        return 0


def index_to_value(index, ref, metric_key):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR         v = r / i
    return index * ref  if metric_key in HIGHER_BETTER else ref / index


def index_to_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries


def calc_accuracy(res, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'top_5_accuracy' if top5 else 'accuracy'
    return res[split]['results']['metrics'][metric]


def calc_parameters(res):
    if 'validation' in res:
        return res['validation']['results']['model']['params'] * 1e-6
    return res['results']['model']['params'] * 1e-6


def calc_gflops(res):
    if 'validation' in res:
        # print(res['config']['backend'], res['execution_platform']['Node Name'], res['validation']['results']['model']['flops'] * 1e-9)
        return res['validation']['results']['model']['flops'] * 1e-9
    return res['results']['model']['flops'] * 1e-9


def calc_fsize(res):
    if 'validation' in res:
        return res['validation']['results']['model']['fsize'] * 1e-6
    return res['results']['model']['fsize'] * 1e-6


def calc_inf_time(res):
    return res['validation']['duration'] / 50000 * 1000


def calc_power_draw(res):
    # TODO add the RAPL measurements if available
    power_draw = 0
    if res['validation']["monitoring_pynvml"] is not None:
        power_draw += res['validation']["monitoring_pynvml"]["total"]["total_power_draw"]
    if res['validation']["monitoring_pyrapl"] is not None:
        power_draw += res['validation']["monitoring_pyrapl"]["total"]["total_power_draw"]
    return power_draw / 50000


def calc_power_draw_train(res, per_epoch=False):
    # TODO add the RAPL measurements if available
    val_per_epoch = res["monitoring_pynvml"]["total"]["total_power_draw"] / len(res["results"]["history"]["loss"])
    val_per_epoch /= 3600000 # Ws to kWh
    if not per_epoch:
        val_per_epoch *= MODEL_INFO[res["config"]["model"]]['epochs']
    return val_per_epoch


def calc_time_train(res, per_epoch=False):
    val_per_epoch = res["duration"] / len(res["results"]["history"]["loss"])
    val_per_epoch /= 3600 # s to h
    if not per_epoch:
        val_per_epoch *= MODEL_INFO[res["config"]["model"]]['epochs']
    return val_per_epoch


def characterize_monitoring(summary):
    sources = {
        'GPU': ['NVML'] if summary['monitoring_pynvml'] is not None else [],
        'CPU': ['RAPL'] if summary['monitoring_pyrapl'] is not None else [],
        'Extern': []
    }
    # TODO also make use of summary['monitoring_psutil']
    # if summary['monitoring_psutil'] is not None:
    #     sources['CPU'].append('psutil')
    return sources


def calculate_optimal_boundaries(summaries, quantiles):
    boundaries = {}
    for task, sum_task in summaries.items():
        for metric in TASK_METRICS_CALCULATION[task].keys():
            index_values = [ env_sum[metric]['index'] for env_sums in sum_task.values() for env_sum in env_sums if env_sum[metric]['index'] is not None ]
            try:
                boundaries[metric] = np.quantile(index_values, quantiles)
            except Exception as e:
                print(e)
    return load_boundaries(boundaries)


def load_boundaries(content="mlee/boundaries.json"):
    if isinstance(content, dict):
        boundary_json = content
    elif isinstance(content, str):
        with open(content, "r") as file:
            boundary_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 10000
    min_value = 0

    boundary_intervals = {}

    for key, boundaries in boundary_json.items():
        intervals = [[max_value, boundaries[0]]]
        for i in range(len(boundaries)-1):
            intervals.append([boundaries[i], boundaries[i+1]])
        intervals.append([boundaries[-1], min_value])
        
        boundary_intervals[key] = intervals

    return boundary_intervals


def save_boundaries(boundary_intervals, output="boundaries.json"):
    scale = {}
    for key in boundary_intervals.keys():
        scale[key] = [sc[0] for sc in boundary_intervals[key][1:]]

    if output is not None:
        with open(output, 'w') as out:
            json.dump(scale, out, indent=4)
    
    return json.dumps(scale, indent=4)


def save_weights(summaries, output="weights.json"):
    weights = {}
    for task_summaries in summaries.values():
        any_summary = list(task_summaries.values())[0][0]
        for key, vals in any_summary.items():
            if isinstance(vals, dict) and 'weight' in vals:
                weights[key] = vals['weight']
    if output is not None:
        with open(output, 'w') as out:
            json.dump(weights, out, indent=4)
    
    return json.dumps(weights, indent=4)


def update_weights(summaries, weights, axis=None):
    for task_summaries in summaries.values():
        for env_summaries in task_summaries.values():
            for model_sum in env_summaries:
                if isinstance(weights, dict):
                    for key, values in model_sum.items():
                        if key in weights:
                            values['weight'] = weights[key]
                else: # only update a single metric weight
                    if axis in model_sum:
                        model_sum[axis]['weight'] = weights
    return summaries


def load_results(results_directory, weighting=None):
    if weighting is None:
        with open(os.path.join(os.path.dirname(__file__), 'weighting.json'), 'r') as wf:
            weighting = json.load(wf)

    logs = {task: {} for task in TASK_TYPES.values()}
    for fname in os.listdir(results_directory):
        with open(os.path.join(results_directory, fname), 'r') as rf:
            log = json.load(rf)
            env_key = get_environment_key(log)
            task_type = TASK_TYPES[fname.split('_')[0]]
            if env_key not in logs[task_type]:
                logs[task_type][env_key] = {}
            if log['config']['model'] in logs[task_type][env_key]:
                raise NotImplementedError(f'Already found results for {log["config"]["model"]} on {env_key}, averaging runs is not implemented (yet)!')
            logs[task_type][env_key][log['config']['model']] = log

    # Exctract all relevant metadata
    summaries = {task: {} for task in TASK_TYPES.values()}
    for task, metrics in TASK_METRICS_CALCULATION.items():
        for env_key, env_logs in logs[task].items():
            if env_key not in summaries[task]:
                summaries[task][env_key] = []
        
            # Calculate inference metrics for rating
            for model_name, model_log in env_logs.items():
                model_information = {
                    'environment': env_key,
                    'name': model_name,
                    'dataset': 'ImageNet',
                    'task_type': task.capitalize(),
                    'power_draw_sources': characterize_monitoring(model_log if 'monitoring_pynvml' in model_log else model_log['validation'])
                }
                for metrics_key, (use_log, metrics_calculation) in metrics.items():
                    try:
                        if not use_log: # calculate based on the inference log
                            model_information[metrics_key] = {'value': metrics_calculation(logs['inference'][env_key][model_name])}
                        else:
                            model_information[metrics_key] = {'value': metrics_calculation(model_log)}
                    except Exception:
                        model_information[metrics_key] = {'value': None}
                    model_information[metrics_key]['weight'] = weighting[metrics_key]
                summaries[task][env_key].append(model_information)

    # Transform logs dict for one environment to list of logs
    for task, task_logs in logs.items():
        for env_key, env_logs in task_logs.items():
            logs[task][env_key] = [model_logs for model_logs in env_logs.values()]

    return logs, summaries


def rate_results(summaries, reference_name, boundaries=None):
    if boundaries is None:
        boundaries = load_boundaries()

    # Get reference values for each environment and task
    reference_values = {}
    for task, task_logs in summaries.items():
        reference_values[task] = {env_key: {} for env_key in task_logs.keys()}
        for env_key, env_logs in task_logs.items():
            for model in env_logs:
                if model['name'] == reference_name:
                    for metrics_key, metrics_val in model.items():
                        if isinstance(metrics_val, dict) and 'value' in metrics_val:
                            if metrics_val['value'] is None:
                                raise RuntimeError(f'Found unratable metric {metrics_key} for reference model {reference_name} on {env_key} {task}!')
                            reference_values[task][env_key][metrics_key] = metrics_val['value']
                    break

    # Calculate value indices using reference values and boundaries
    for task, task_summs in summaries.items():
        for env_key, env_summs in task_summs.items():
            for model in env_summs:
                for key in model.keys():
                    if isinstance(model[key], dict) and 'value' in model[key]:
                        if model[key]['value'] is None:
                            model[key]['index'] = None
                            model[key]['rating'] = 4
                        else:
                            model[key]['index'] = value_to_index(model[key]['value'], reference_values[task][env_key][key], key)
                            model[key]['rating'] = index_to_rating(model[key]['index'], boundaries[key])

    # Calculate the real-valued boundaries
    real_boundaries = {}
    for task, task_ref_values in reference_values.items():
        real_boundaries[task] = {env_key: {} for env_key in task_ref_values.keys()}
        for env_key, env_ref_values in task_ref_values.items():
            for key, val in env_ref_values.items():
                real_boundaries[task][env_key][key] = [(index_to_value(start, val, key), index_to_value(stop, val, key)) for (start, stop) in boundaries[key]]
    
    return summaries, boundaries, real_boundaries
