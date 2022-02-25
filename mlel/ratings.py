import os
import json
from re import T

import numpy as np


HIGHER_BETTER = [
    'top1_val',
    'top5_val',
]
BACKENDS = {
    'tensorflow': ('TensorFlow', 'tensorflow'),
    'pytorch': ('Torch', 'torch'),
}
GPU_NAMES = {
    'NVIDIA A100-SXM4-40GB': 'A100',
    'Quadro RTX 5000': 'RTX 5000',
}
FULL_TRAIN_EPOCHS = {
    'ResNet50':             90, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet101':            90, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNet152':            90, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG16':                90, # https://github.com/pytorch/vision/tree/main/references/classification
    'VGG19':                90, # https://github.com/pytorch/vision/tree/main/references/classification
    'EfficientNetB0':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB1':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB2':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB3':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB4':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB5':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB6':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'EfficientNetB7':       None, # no information found https://arxiv.org/pdf/1905.11946.pdf
    'RegNetX400MF':         100, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX32GF':          100, # https://github.com/pytorch/vision/tree/main/references/classification
    'RegNetX8GF':           100, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext50':            100, # https://github.com/pytorch/vision/tree/main/references/classification
    'ResNext101':           100, # https://github.com/pytorch/vision/tree/main/references/classification
    'DenseNet121':          90, # Densely Connected Convolutional Networks https://arxiv.org/pdf/1608.06993.pdf
    'DenseNet169':          90, # Densely Connected Convolutional Networks https://arxiv.org/pdf/1608.06993.pdf
    'DenseNet201':          90, # Densely Connected Convolutional Networks https://arxiv.org/pdf/1608.06993.pdf
    'Xception':             None, # no information found
    'InceptionResNetV2':    200, # Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning https://arxiv.org/pdf/1602.07261.pdf
    'InceptionV3':          100, # Rethinking the Inception Architecture for Computer Vision https://arxiv.org/pdf/1512.00567.pdf
    'NASNetMobile':         100, # Learning Transferable Architectures for Scalable Image Recognition https://arxiv.org/pdf/1707.07012.pdf
    'MobileNetV2':          300, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Small':     600, # https://github.com/pytorch/vision/tree/main/references/classification
    'MobileNetV3Large':     600, # https://github.com/pytorch/vision/tree/main/references/classification
    'QuickNetSmall':        600, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNet':             600, # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
    'QuickNetLarge':        600 # https://github.com/larq/zoo/blob/main/larq_zoo/training/sota_experiments.py
}
TASK_TYPES = {
    'eval': 'inference',
    'train': 'training',
}
TASK_METRICS_CALCULATION = {        # boolean informs whether given task log is used (True), or if the corresponding inference log shall be used instead
    'inference': {
        'parameters':               (True,  lambda model_log: calc_parameters(model_log)),
        'fsize':                    (True,  lambda model_log: calc_fsize(model_log)),
        'inference_power_draw':     (True,  lambda model_log: calc_power_draw(model_log)),
        'inference_time':           (True,  lambda model_log: calc_inf_time(model_log)),
        'top1_val':                 (True,  lambda model_log: calc_accuracy(model_log)),
        'top5_val':                 (True,  lambda model_log: calc_accuracy(model_log, top5=True))
    },
    'training': {
        'parameters':               (True,  lambda model_log: calc_parameters(model_log)),
        'fsize':                    (True,  lambda model_log: calc_fsize(model_log)),
        'train_power_draw_epoch':   (True,  lambda model_log: calc_power_draw_train(model_log, True)),
        'train_power_draw':         (True,  lambda model_log: calc_power_draw_train(model_log)),
        'train_time_epoch':         (True,  lambda model_log: calc_time_train(model_log, True)),
        'train_time':               (True,  lambda model_log: calc_time_train(model_log)),
        'top1_val':                 (False, lambda model_val_log: calc_accuracy(model_val_log)),
        'top5_val':                 (False, lambda model_val_log: calc_accuracy(model_val_log, top5=True))
    }
}
METRICS_FOR_FINAL_RATING = {        # boolean informs whether given task log is used (True), or if the corresponding inference log shall be used instead
    'inference': ['parameters', 'inference_power_draw', 'inference_time', 'top1_val'],
    'training': ['parameters', 'train_power_draw', 'train_time', 'top1_val']
}


def get_environment_key(log):
    backend_name, pip_name = BACKENDS[log['config']['backend']]
    backend_version = [r.split('==')[1] for r in log['requirements'] if r.split('==')[0] == pip_name][0]
    n_gpus = len(log['execution_platform']['GPU'])
    gpu_name = GPU_NAMES[log['execution_platform']['GPU']['0']['Name']]
    gpu_str = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    return f'{gpu_str} - {backend_name} {backend_version}'


def aggregate_rating(ratings, mode, meanings=None):
    if isinstance(ratings, dict): # model summary given instead of list of ratings
        ratings = [val['rating'] for key, val in ratings.items() if key in METRICS_FOR_FINAL_RATING[ratings['task_type'].lower()]]
    if meanings is None:
        meanings = np.arange(np.max(ratings) + 1)
    round_m = np.ceil if 'pessimistic' in mode else np.floor # optimistic
    if mode == 'best':
        return meanings[min(ratings)]
    if mode == 'worst':
        return meanings[max(ratings)]
    if 'median' in mode:
        return meanings[int(round_m(np.median(ratings)))]
    if 'mean' in mode:
        return meanings[int(round_m(np.mean(ratings)))]
    if mode == 'majority':
        return meanings[np.argmax(np.bincount(ratings))]
    raise NotImplementedError('Rating Mode not implemented!', mode)


def value_to_index(value, ref, metric_key):
    # TODO If values is integer, just return integer
    #      i = v / r                     OR                i = r / v
    return value / ref if metric_key in HIGHER_BETTER else ref / value


def index_to_value(index, ref, metric_key):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR          v = r / i
    return index * ref  if metric_key in HIGHER_BETTER else ref / index


def calculate_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i


def calc_accuracy(res, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'top_5_accuracy' if top5 else 'accuracy'
    return res[split]['results']['metrics'][metric]


def calc_parameters(res):
    if 'validation' in res:
        return res['validation']['results']['model']['params'] * 1e-6
    return res['results']['model']['params'] * 1e-6


def calc_fsize(res):
    if 'validation' in res:
        return res['validation']['results']['model']['fsize']
    return res['results']['model']['fsize']


def calc_inf_time(res):
    return res['train']['duration'] / 1281167 * 1000


def calc_power_draw(res):
    # TODO add the RAPL measurements if available
    return res['train']["monitoring_gpu"]["total"]["total_power_draw"] / 1281167


def calc_power_draw_train(res, per_epoch=False):
    # TODO add the RAPL measurements if available
    val_per_epoch = res["monitoring_gpu"]["total"]["total_power_draw"] / len(res["results"]["history"]["loss"])
    val_per_epoch /= 3600000 # Ws to kWh
    if not per_epoch:
        val_per_epoch *= FULL_TRAIN_EPOCHS[res["config"]["model"]]
    return val_per_epoch


def calc_time_train(res, per_epoch=False):
    val_per_epoch = res["duration"] / len(res["results"]["history"]["loss"])
    val_per_epoch /= 3600 # s to h
    if not per_epoch:
        val_per_epoch *= FULL_TRAIN_EPOCHS[res["config"]["model"]]
    return val_per_epoch


def characterize_monitoring(summary):
    sources = {
        'GPU': ['NVML'] if summary['monitoring_gpu'] is not None else [],
        'CPU': ['RAPL'] if summary['monitoring_rapl'] is not None else [],
        'Extern': []
    }
    # TODO also make use of summary['monitoring_cpu']
    # if summary['monitoring_cpu'] is not None:
    #     sources['CPU'].append('psutil')
    return sources


def load_scale(content="mlel/scales.json"):
    if isinstance(content, dict):
        scales_json = content
    elif isinstance(content, str):
        with open(content, "r") as file:
            scales_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 100
    min_value = 0

    scale_intervals = {}

    for key, boundaries in scales_json.items():
        intervals = [[max_value, boundaries[0]]]
        for i in range(len(boundaries)-1):
            intervals.append([boundaries[i], boundaries[i+1]])
        intervals.append([boundaries[-1], min_value])
        
        scale_intervals[key] = intervals

    return scale_intervals


def save_scale(scale_intervals, output="scales.json"):
    scale = {}
    for key in scale_intervals.keys():
        scale[key] = [sc[0] for sc in scale_intervals[key][1:]]

    if output is not None:
        with open(output, 'w') as out:
            json.dump(scale, out, indent=4)
    
    return json.dumps(scale, indent=4)


def load_results(results_directory):
    logs = {}
    for fname in os.listdir(results_directory):
        with open(os.path.join(results_directory, fname), 'r') as rf:
            log = json.load(rf)
            env_key = get_environment_key(log)
            if env_key not in logs:
                logs[env_key] = {task: {} for task in TASK_TYPES.values()}
            task_type = TASK_TYPES[fname.split('_')[0]]
            if log['config']['model'] in logs[env_key][task_type]:
                raise NotImplementedError(f'Already found results for {log["config"]["model"]} on {env_key}, averaging runs is not implemented (yet)!')
            logs[env_key][task_type][log['config']['model']] = log

    # Exctract all relevant metadata
    summaries = {}
    for env_key, env_logs in logs.items():
        if env_key not in summaries:
            summaries[env_key] = {task: [] for task in TASK_TYPES.values()}
        
        # Calculate inference metrics for rating
        for task, metrics in TASK_METRICS_CALCULATION.items():
            for model_name, model_log in env_logs[task].items():
                model_information = {
                    'environment': env_key,
                    'name': model_name,
                    'dataset': 'ImageNet',
                    'task_type': task.capitalize(),
                    'power_draw_sources': characterize_monitoring(model_log if 'monitoring_gpu' in model_log else model_log['train'])
                }
                for metrics_key, (use_log, metrics_calculation) in metrics.items():
                    try:
                        if not use_log: # calculate based on the inference log
                            model_information[metrics_key] = {'value': metrics_calculation(logs[env_key]['inference'][model_name])}
                        else:
                            model_information[metrics_key] = {'value': metrics_calculation(model_log)}
                    except Exception:
                        model_information[metrics_key] = {'value': None}
                summaries[env_key][task].append(model_information)

    # Transform logs dict for one environment to list of logs
    for env_key, env_logs in logs.items():
        logs[env_key]['inference'] = [model_logs for model_logs in env_logs['inference'].values()]
        logs[env_key]['training'] = [model_logs for model_logs in env_logs['training'].values()]

    return logs, summaries


def rate_results(summaries, reference_name, scales=None):
    if scales is None:
        scales = load_scale()

    # Get reference values for each environment and task
    reference_values = {}
    for env_key, env_logs in summaries.items():
        type_ref_values = {task_type: {} for task_type in TASK_TYPES.values()}
        for task_type, type_logs in env_logs.items():
            for model in type_logs:
                if model['name'] == reference_name:
                    for metrics_key, metrics_val in model.items():
                        if isinstance(metrics_val, dict) and 'value' in metrics_val:
                            if metrics_val['value'] is None:
                                raise RuntimeError(f'Found unratable metric {metrics_key} for reference model {reference_name} on {env_key} {task_type}!')
                            type_ref_values[task_type][metrics_key] = metrics_val['value']
                    break
        reference_values[env_key] = type_ref_values

    # Calculate value indices using reference values and scales
    for env_key, env_logs in summaries.items():
        for task_type, type_logs in env_logs.items():
            for model in type_logs:
                for key in model.keys():
                    if isinstance(model[key], dict) and 'value' in model[key]:
                        if model[key]['value'] is None:
                            model[key]['index'] = None
                            model[key]['rating'] = 4
                        else:
                            model[key]['index'] = value_to_index(model[key]['value'], reference_values[env_key][task_type][key], key)
                            model[key]['rating'] = calculate_rating(model[key]['index'], scales[key])

    # Calculate the real-valued scales
    real_scales = {}
    for env_key, ref_values in reference_values.items():
        real_scales[env_key] = {'inference': {}, 'training': {}}
        for task_type, type_ref_values in ref_values.items():
            for key, val in type_ref_values.items():
                real_scales[env_key][key] = [(index_to_value(start, val, key), index_to_value(stop, val, key)) for (start, stop) in scales[key]]
    
    return summaries, scales, real_scales
