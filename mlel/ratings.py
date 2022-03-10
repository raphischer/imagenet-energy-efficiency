import os
import json

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
METRICS_FOR_FINAL_RATING = {
    'inference': ['parameters', 'inference_power_draw', 'inference_time', 'top1_val'],
    'training': ['parameters', 'train_power_draw', 'train_time', 'top1_val']
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
            if req.split('==')[0] == package:
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
    #      i = v / r                     OR                i = r / v
    return value / ref if metric_key in HIGHER_BETTER else ref / value


def index_to_value(index, ref, metric_key):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR         v = r / i
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


def calc_gflops(res):
    if 'validation' in res:
        # print(res['config']['backend'], res['execution_platform']['Node Name'], res['validation']['results']['model']['flops'] * 1e-9)
        return res['validation']['results']['model']['flops'] * 1e-9
    return res['results']['model']['flops'] * 1e-9


def calc_fsize(res):
    if 'validation' in res:
        return res['validation']['results']['model']['fsize']
    return res['results']['model']['fsize']


def calc_inf_time(res):
    return res['validation']['duration'] / ((50000 // res['config']['batch_size'])) * 1000


def calc_power_draw(res):
    # TODO add the RAPL measurements if available
    power_draw = 0
    if res['validation']["monitoring_pynvml"] is not None:
        power_draw += res['validation']["monitoring_pynvml"]["total"]["total_power_draw"]
    if res['validation']["monitoring_pyrapl"] is not None:
        power_draw += res['validation']["monitoring_pyrapl"]["total"]["total_power_draw"]
    return power_draw / (50000 // res['config']['batch_size'])


def calc_power_draw_train(res, per_epoch=False):
    # TODO add the RAPL measurements if available
    val_per_epoch = res["monitoring_pynvml"]["total"]["total_power_draw"] / len(res["results"]["history"]["loss"])
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
        'GPU': ['NVML'] if summary['monitoring_pynvml'] is not None else [],
        'CPU': ['RAPL'] if summary['monitoring_pyrapl'] is not None else [],
        'Extern': []
    }
    # TODO also make use of summary['monitoring_psutil']
    # if summary['monitoring_psutil'] is not None:
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
                summaries[task][env_key].append(model_information)

    # Transform logs dict for one environment to list of logs
    for task, task_logs in logs.items():
        for env_key, env_logs in task_logs.items():
            logs[task][env_key] = [model_logs for model_logs in env_logs.values()]

    return logs, summaries


def rate_results(summaries, reference_name, scales=None):
    if scales is None:
        scales = load_scale()

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

    # Calculate value indices using reference values and scales
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
                            model[key]['rating'] = calculate_rating(model[key]['index'], scales[key])

    # Calculate the real-valued scales
    real_scales = {}
    for task, task_ref_values in reference_values.items():
        real_scales[task] = {env_key: {} for env_key in task_ref_values.keys()}
        for env_key, env_ref_values in task_ref_values.items():
            for key, val in env_ref_values.items():
                real_scales[task][env_key] = [(index_to_value(start, val, key), index_to_value(stop, val, key)) for (start, stop) in scales[key]]
    
    return summaries, scales, real_scales
