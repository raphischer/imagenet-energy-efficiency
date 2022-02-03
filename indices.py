import json
from results_vis import calc_accuracy, calc_inf_time, calc_power_draw, calc_fsize, calc_parameters

keys = ["parameters", "fsize", "power_draw", "inference_time", "top1-val", "top5-val"]

def load_scale(path="scales.json"):
    with open(path, "r") as file:
        scales_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 1e5
    min_value = 1e-5

    scale_intervals = {}

    for key in keys:
        boundaries = scales_json[key]
        intervals = [(max_value, boundaries[0])]
        for i in range(len(boundaries)-1):
            intervals.append((boundaries[i], boundaries[i+1]))
        intervals.append([(boundaries[-1], min_value)])
        
        scale_intervals[key] = intervals

    return scale_intervals

if __name__ == "__main__":
    result_files = {
        # 'A100_Tensorflow': 'results/A100/results_tf_pretrained.json',
        'A100_PyTorch': 'results/A100/results_torch_pretrained.json',
        # 'RTX5000_Tensorflow': 'results/RTX5000/results_tf_pretrained.json',
        # 'RTX5000_PyTorch': 'results/RTX5000/results_torch_pretrained.json',
    }
    results = {}
    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            results[name] = json.load(r)

    agg = {}
    for k in results["A100_PyTorch"].keys():
        r = results["A100_PyTorch"][k]

        model_name = r["config"]["model"]
        model_values = {
            "parameters": calc_parameters(r),
            "fsize": calc_fsize(r),
            "power_draw": calc_power_draw(r),
            "inference_time": calc_inf_time(r),
            "top1-val": calc_accuracy(r),
            "top5-val": calc_accuracy(r, top5=True),
        }
        
        agg[model_name] = model_values

    # Default model: ResNet101

    comparators = {
        # Smaller is better
        "parameters": lambda ref, v: ref / v,
        "fsize": lambda ref, v: ref / v,
        "power_draw": lambda ref, v: ref / v,
        # Bigger is better
        "inference_time": lambda ref, v: v / ref,
        "top1-val": lambda ref, v: v / ref,
        "top5-val": lambda ref, v: v / ref,
    }

    scales = load_scale()
    scale_representation = ["++", "+", "o", "-", "--"]

    reference_values = agg['ResNet101']
    del agg['ResNet101']

    for param_name, comp_fn in comparators.items():
        print("-"*100)
        print(param_name)
        print("-"*100)
        for model_name, value_array in agg.items():
            index = comp_fn(reference_values[param_name], value_array[param_name])
            rating = ""
            try:
                s = scales[param_name]
                for i, (upper, lower) in enumerate(s):
                    if index <= upper and index > lower:
                        rating = scale_representation[i]
                        break
            except Exception:
                pass

            print(model_name, index, rating)
        
        print("")
