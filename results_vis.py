import json

import numpy as np

with open('plots/eval_a100/results_eval.json', 'r') as r:
    res_a100 = json.load(r)
with open('plots/eval_rtx5000/results_eval.json', 'r') as r:
    res_rtx = json.load(r)

print(1)
for mod, res in res_a100.items():
    if mod in res_rtx:
        valid_acc_a100 = res['validation']['results']['metrics']['accuracy']
        valid_acc_rtx = res_rtx[mod]['validation']['results']['metrics']['accuracy']
        train_acc_a100 = res['train']['results']['metrics']['accuracy']
        train_acc_rtx = res_rtx[mod]['train']['results']['metrics']['accuracy']

        train_acc_diff = np.abs((train_acc_a100 - train_acc_rtx) * 100)
        valid_acc_diff = np.abs((valid_acc_a100 - valid_acc_rtx) * 100)

        print(f'{mod:<30}, valid acc diff {valid_acc_diff:5.3f} %, train acc diff {train_acc_diff:5.3f} %')

        a = 1
