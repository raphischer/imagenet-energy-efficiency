import os

import numpy as np

from load_imagenet import NUM_SAMPLES


def calculate_bar_width(nr_bars, desired_padding):
    space_after_padding = 1 - 2 * desired_padding
    width = space_after_padding / nr_bars
    return width


def calculate_x_positions(nr_bars, width):
    nr_one_side = nr_bars // 2
    odd_nr_bars = nr_bars % 2 == 1
    offsets = [i * width/2 for i in range(nr_one_side)]
    one_side = np.array([offsets[i-1] + i * width/2 + odd_nr_bars*width/2 for i in range(1, nr_one_side+1)])

    if odd_nr_bars:
        return np.concatenate([np.flip(-one_side), np.zeros((1)), one_side])
    else:
        return np.concatenate([np.flip(-one_side), one_side])


def grouped_barplot(ax, ys, x_positions=None, padding=0.1, colors=None, labels=None):
    nr_ticks, bars_per_tick = ys.shape
    if x_positions is None:
        x_positions = np.arange(nr_ticks)
    if colors is None:
        colors = [f"C{i}" for i in range(bars_per_tick)]

    used_labels = set()
    for x, y in zip(x_positions, ys):
        nr_bars = len(y)
        width = calculate_bar_width(nr_bars, padding)
        xs = calculate_x_positions(nr_bars, width) + x
        for i, (_x, _y) in enumerate(zip(xs,y)):
            if labels is not None and labels[i] not in used_labels:
                ax.bar(x=_x, height=_y, width=width, color=colors[i], label=labels[i])
                used_labels.add(labels[i])
            else:
                ax.bar(x=_x, height=_y, width=width, color=colors[i])


def create_train_plots(results, directory):
    import matplotlib.pyplot as plt

    for dir, res in results.items():
        history = res['results']['history']
        train_acc = history['accuracy']
        valid_acc = history['val_accuracy']
        epochs = list(range(len(train_acc)))
        plt.clf()
        plt.plot(epochs, train_acc, label='Training')
        plt.plot(epochs, valid_acc, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')
        plt.legend()
        plt.savefig(os.path.join(directory, f'conv_{dir}.png'), bbox_inches="tight")
    
    names = list(training_results.keys())

    fsizes = [val['results']['model']['fsize'] * 1e-6 for val in training_results.values()]
    params = [val['results']['model']['params'] * 1e-6 for val in training_results.values()]
    
    power_draw = [list(val["gpu_monitoring"].values())[0]["total_power_draw"] / 3600000 for val in training_results.values()]
    rel_power_draw = [list(val["gpu_monitoring"].values())[0]["total_power_draw"] / (val['epochs'] * 3600) for val in training_results.values()]
    train_acc = [val['results']['metrics']['accuracy'] * 100 for val in eval_train_results.values()]
    valid_acc = [val['results']['metrics']['accuracy'] * 100 for val in eval_valid_results.values()]

    plt.clf()
    plt.scatter(fsizes, params)
    for i, txt in enumerate(names):
        plt.annotate(txt, (fsizes[i], params[i]))
    plt.xlabel('File size (MB)')
    plt.ylabel('N Params (M)')
    plt.savefig(os.path.join(directory, 'sizes_correlation.png'), bbox_inches="tight")

    plt.clf()
    plt.bar(names, params)
    plt.xticks(rotation=45)
    plt.ylabel('N Params (M)')
    plt.savefig(os.path.join(directory, 'sizes_params.png'), bbox_inches="tight")

    plt.clf()
    plt.bar(names, power_draw)
    plt.xticks(rotation=45)
    plt.ylabel('Power Draw (kWh)')
    plt.savefig(os.path.join(directory, 'power_draw.png'), bbox_inches="tight")

    plt.clf()
    plt.bar(names, rel_power_draw)
    plt.xticks(rotation=45)
    plt.ylabel('Rel Power Draw (Wh / Epoch)')
    plt.savefig(os.path.join(directory, 'power_draw_rel.png'), bbox_inches="tight")

    plt.clf()
    plt.scatter(params, rel_power_draw)
    for i, txt in enumerate(names):
        plt.annotate(txt, (params[i], rel_power_draw[i]))
    plt.xlabel('N Params (M)')
    plt.ylabel('Rel Power Draw (Wh / Epoch)')
    plt.savefig(os.path.join(directory, 'power_draw_rel_correlation_size.png'), bbox_inches="tight")

    plt.clf()
    plt.bar([x - 0.2 for x in range(len(names))], train_acc, width=0.3, tick_label=names, label='train')
    plt.bar([x + 0.2 for x in range(len(names))], valid_acc, width=0.3, tick_label=names, label='valid')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.savefig(os.path.join(directory, 'accuracy.png'), bbox_inches="tight")

    plt.clf()
    plt.scatter(valid_acc, params)
    for i, txt in enumerate(names):
        plt.annotate(txt, (valid_acc[i], params[i]))
    plt.xlabel('Valid accuracy [%]')
    plt.ylabel('N Params (M)')
    plt.savefig(os.path.join(directory, 'accuracy_correlation_size.png'), bbox_inches="tight")

    plt.clf()
    plt.scatter(valid_acc, power_draw)
    for i, txt in enumerate(names):
        plt.annotate(txt, (valid_acc[i], power_draw[i]))
    plt.xlabel('Valid accuracy [%]')
    plt.ylabel('Power Draw (kWh)')
    plt.savefig(os.path.join(directory, 'accuracy_correlation_power.png'), bbox_inches="tight")

    plt.clf()
    plt.scatter(valid_acc, rel_power_draw)
    for i, txt in enumerate(names):
        plt.annotate(txt, (valid_acc[i], rel_power_draw[i]))
    plt.xlabel('Valid accuracy [%]')
    plt.ylabel('Rel Power Draw (Wh / epoch)')
    plt.savefig(os.path.join(directory, 'accuracy_correlation_rel_power.png'), bbox_inches="tight")

    print(1)
    

def create_evaluation_plots(results, directory):
    import matplotlib.pyplot as plt

    # training_results = {val['training']['model']: val['training'] for val in results.values()}
    eval_train_results = {val['train']['model']: val['train'] for val in results.values()}
    eval_valid_results = {val['validation']['model']: val['validation'] for val in results.values()}

    names = list(eval_train_results.keys())
    
    train_power_draw = [val["gpu_monitoring"]["total"]["total_power_draw"] / 3600 for val in eval_train_results.values()]
    valid_power_draw = [val["gpu_monitoring"]["total"]["total_power_draw"] / 3600 for val in eval_valid_results.values()]
    rel_train_power_draw = [draw * 3600 / NUM_SAMPLES['train'] for draw in train_power_draw]
    rel_valid_power_draw = [draw * 3600 / NUM_SAMPLES['validation'] for draw in valid_power_draw]
    train_acc = [val['results']['metrics']['accuracy'] * 100 for val in eval_train_results.values()]
    valid_acc = [val['results']['metrics']['accuracy'] * 100 for val in eval_valid_results.values()]

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    grouped_barplot(ax, np.array([np.array(train_power_draw), np.array(valid_power_draw)]).transpose(), labels=['Training', 'Validations'])
    ax.set_xticklabels(names)
    fig.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Power Draw [Wh]')
    plt.savefig(os.path.join(directory, 'power_draw.png'), bbox_inches="tight")

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    grouped_barplot(ax, np.array([np.array(rel_train_power_draw), np.array(rel_valid_power_draw)]).transpose(), labels=['Training', 'Validations'])
    ax.set_xticklabels(names)
    fig.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Power Draw [Ws / sample]')
    plt.savefig(os.path.join(directory, 'power_draw_rel.png'), bbox_inches="tight")

    plt.clf()
    fig, ax = plt.subplots(1, 1)
    grouped_barplot(ax, np.array([np.array(train_acc), np.array(valid_acc)]).transpose(), labels=['Training', 'Validations'])
    ax.set_xticklabels(names)
    fig.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy [%]')
    plt.savefig(os.path.join(directory, 'accuracy.png'), bbox_inches="tight")

    plt.clf()
    plt.scatter(train_acc, rel_train_power_draw)
    for i, txt in enumerate(names):
        plt.annotate(txt, (train_acc[i], rel_train_power_draw[i]))
    plt.xlabel('Train accuracy [%]')
    plt.ylabel('Power Draw [Ws / sample]')
    plt.savefig(os.path.join(directory, 'corr_train_accuracy_rel_power.png'), bbox_inches="tight")

    plt.clf()
    plt.scatter(valid_acc, rel_valid_power_draw)
    for i, txt in enumerate(names):
        plt.annotate(txt, (valid_acc[i], rel_valid_power_draw[i]))
    plt.xlabel('Validation accuracy [%]')
    plt.ylabel('Power Draw [Ws / sample]')
    plt.savefig(os.path.join(directory, 'corr_valid_accuracy_rel_power.png'), bbox_inches="tight")
