import os

def create_plots(results, directory):
    import matplotlib.pyplot as plt

    results = {key: res for key, res in results.items() if not res['training']['use_simple']} # TODO remove experiments with simple prepr

    training_results = {val['training']['model']: val['training'] for val in results.values()}
    eval_train_results = {val['train']['model']: val['train'] for val in results.values()}
    eval_valid_results = {val['validation']['model']: val['validation'] for val in results.values()}
    names = list(training_results.keys())
    fsizes = [val['model_fsize'] * 1e-6 for val in training_results.values()]
    params = [val['model_params'] * 1e-6 for val in training_results.values()]
    train_time = [val['duration'] for val in training_results.values()]
    epochs = [val['epochs'] for val in training_results.values()]
    power_draw = [list(val["gpu_monitoring"].values())[0]["total_power_draw"] / 3600000 for val in training_results.values()]
    rel_power_draw = [list(val["gpu_monitoring"].values())[0]["total_power_draw"] / (val['epochs'] * 3600) for val in training_results.values()]
    train_acc = [val['eval_metrics']['accuracy'] * 100 for val in eval_train_results.values()]
    valid_acc = [val['eval_metrics']['accuracy'] * 100 for val in eval_valid_results.values()]

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
    