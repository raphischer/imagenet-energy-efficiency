import json

KEYS = ["parameters", "fsize", "power_draw", "inference_time", "top1_val", "top5_val"]
RATINGS = ['green', 'yellow', 'orange', 'red', 'gray']
HIGHER_BETTER = [
    'top1_val',
    'top5_val',
]

def calculate_index(values, ref, axis):
    # TODO If values is integer, just return integer
    return [val / ref for val in values] if axis in HIGHER_BETTER else [ref / val for val in values]


def calculate_rating(values, scale):
    ratings = []
    for index in values:
        for i, (upper, lower) in enumerate(scale):
            if index <= upper and index > lower:
                ratings.append(i)
                break
    return ratings


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
    max_value = 1e5
    min_value = 1e-5

    scale_intervals = {}

    for key in KEYS:
        boundaries = scales_json[key]
        intervals = [(max_value, boundaries[0])]
        for i in range(len(boundaries)-1):
            intervals.append((boundaries[i], boundaries[i+1]))
        intervals.append((boundaries[-1], min_value))
        
        scale_intervals[key] = intervals

    return scale_intervals


def load_results(result_files):
    tmp = {}
    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            tmp[name] = json.load(r)

    scales = load_scale()

    # Exctract all relevant metadata
    results = {}
    for exp_name, resf in tmp.items():
        for model in resf.values():
            model_information = {}
            model_information['name'] = model['config']['model']
            model_information['parameters'] = calc_parameters(model)
            model_information['fsize'] = calc_fsize(model)
            model_information['power_draw'] = calc_power_draw(model)
            model_information['inference_time'] = calc_inf_time(model)
            model_information['top1_val'] = calc_accuracy(model)
            model_information['top5_val'] = calc_accuracy(model, top5=True)

            try:
                results[exp_name].append(model_information)
            except Exception:
                results[exp_name] = [model_information]


    # Get reference values
    reference_values = {}
    for exp_name, model_list in results.items():
        for model in model_list:
            if model['name'] == 'ResNet101':
                reference_values[exp_name] = {k: v for k, v in model.items() if k != 'name'}
                break

    # Calculate indices using reference values and scales
    for exp_name, model_list in results.items():
        for model in model_list:
            model['indices'] = {}

            for key in KEYS:
                index = calculate_index([model[key]], reference_values[exp_name][key], key)[0]
                model['indices'][key] = {'value': index, 'rating': calculate_rating([index], scales[key])[0] }
    
    return results
