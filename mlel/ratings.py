import json

import numpy as np


KEYS = ["parameters", "fsize", "power_draw", "inference_time", "top1_val", "top5_val"]
HIGHER_BETTER = [
    'top1_val',
    'top5_val',
]


def aggregate_rating(ratings, mode, meanings=None):
    if meanings is None:
        meanings = np.arange(np.max(ratings) + 1)
    if mode == 'best':
        return meanings[min(ratings)]
    if mode == 'worst':
        return meanings[max(ratings)]
    if mode == 'median':
        return meanings[int(np.median(ratings))]
    if mode == 'mean':
        return meanings[int(np.ceil(np.mean(ratings)))]
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
    return res['validation']['results']['model']['params']


def calc_fsize(res):
    return res['validation']['results']['model']['fsize']


def calc_inf_time(res):
    return res['train']['duration'] / 1281167 * 1000


def calc_power_draw(res):
    return res['train']["monitoring_gpu"]["total"]["total_power_draw"] / 1281167


def load_scale(path="mlel/scales.json"):
    with open(path, "r") as file:
        scales_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 100
    min_value = 0

    scale_intervals = {}

    for key in KEYS:
        boundaries = scales_json[key]
        intervals = [(max_value, boundaries[0])]
        for i in range(len(boundaries)-1):
            intervals.append((boundaries[i], boundaries[i+1]))
        intervals.append((boundaries[-1], min_value))
        
        scale_intervals[key] = intervals

    return scale_intervals


def rate_results(result_files, scales, reference_name):
    tmp = {}

    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            tmp[name] = json.load(r)

    # Exctract all relevant metadata
    results = {}
    for exp_name, resf in tmp.items():
        for model in resf.values():
            model_information = {}
            model_information['name'] = model['config']['model']
            model_information['parameters'] = {'value': calc_parameters(model)}
            model_information['fsize'] = {'value': calc_fsize(model)}
            model_information['power_draw'] = {'value': calc_power_draw(model)}
            model_information['inference_time'] = {'value': calc_inf_time(model)}
            model_information['top1_val'] = {'value': calc_accuracy(model)}
            model_information['top5_val'] = {'value': calc_accuracy(model, top5=True)}

            try:
                results[exp_name].append(model_information)
            except Exception:
                results[exp_name] = [model_information]

    # Get reference values
    reference_values = {}
    for exp_name, model_list in results.items():
        for model in model_list:
            if model['name'] == reference_name:
                reference_values[exp_name] = {k: v['value'] for k, v in model.items() if k != 'name'}
                break

    # Calculate indices using reference values and scales
    for exp_name, model_list in results.items():
        for model in model_list:
            for key in KEYS:
                model[key]['index'] = value_to_index(model[key]['value'], reference_values[exp_name][key], key)
                model[key]['rating'] = calculate_rating(model[key]['index'], scales[key])

    # Calculate the real-valued scales
    real_scales = {}
    for env, ref_values in reference_values.items():
        real_scales[env] = {}
        for key, vals in scales.items():
            real_scales[env][key] = [(index_to_value(start, ref_values[key], key), index_to_value(stop, ref_values[key], key)) for (start, stop) in vals]
    
    return results, real_scales
