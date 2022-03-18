# importing package
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from mlel.ratings import load_results, rate_results, calculate_compound_rating, load_backend_info
from mlel.label_generator import EnergyLabel

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOGS, summaries = load_results('results')
SUMMARIES, SCALES, SCALES_REAL = rate_results(summaries, 'ResNet101')
RATING_MODE = 'optimistic median'
RATINGS = ['A', 'B', 'C', 'D', 'E']
RA = '#639B30'
RB = '#B8AC2B'
RC = '#F8B830'
RD = '#EF7D29'
RE = '#E52421'
COLORS = [RA, RB, RC, RD, RE]
HATCHES = [ "+" , "\\", "o", "-", "x", "*", "/" , "|", ".", "O" ]
MARKERS = ["," , "o" , "v" , "^" , "<", ">"]
THREE_ENVS = ['A100 x8 - TensorFlow 2.8.0', 'RTX 5000 - TensorFlow 2.4.1', 'Xeon(R) W-2155 - ONNX PT Export 1.10.0']

os.chdir(os.path.dirname(__file__))


##########################################
##### TABLE WITH INDEXING RESULTS ########
##########################################

TEX_TABLE_INDEXING_RESULTS = r'''\resizebox{\linewidth}{!}{
    \begin{tabular}{l||c|c||c|c||c|c||c|c}
        \toprule 
        \multirow{2}{*}{Model (ImageNet)}  & \multicolumn{2}{c}{$METR1} & \multicolumn{2}{c}{$METR2} & \multicolumn{2}{c}{$METR3} & \multicolumn{2}{c|}{$METR4}  \\  \cline{2-9}
        & Value & Index & Value & Index & Value & Index & Value & Index \\ 
        \midrule
        \multicolumn{9}{l}{$ENV1} \\
        $RES1
        \midrule
        \multicolumn{9}{l}{$ENV2} \\
        $RES2
        \midrule
        \multicolumn{9}{l}{$ENV3} \\
        $RES3
        \bottomrule
    \end{tabular}
    }'''
TABLE_NAMES = {
    "parameters": "Parameters [#]",
    "inference_power_draw": "Power Draw [Ws]",
    "inference_time": "Inference Time [ms]",
    "top1_val": "Top-1 Accuracy [%]"
}
res = TEX_TABLE_INDEXING_RESULTS
for e_i, env in enumerate(THREE_ENVS):
    res = res.replace(f'$ENV{e_i + 1}', f'{env} environment')
    env_res = []
    for model in ['ResNet101', 'EfficientNetB0', 'VGG19', 'MobileNetV2']:
        model_res = [model]
        for mod_sum in SUMMARIES['inference'][env]:
            if mod_sum['name'] == model:
                for m_i, metr in enumerate(['inference_power_draw', 'inference_time', 'parameters', 'top1_val']):
                    res = res.replace(f'$METR{m_i + 1}', TABLE_NAMES[metr])
                    ind = r'\colorbox{R' + RATINGS[mod_sum[metr]['rating']] + r'}{' + f"{mod_sum[metr]['index']:4.3f}" + '}'
                    model_res.extend([f"{mod_sum[metr]['value']:4.3f}", ind])
                break
        env_res.append(' & '.join(model_res) + r' \\')
    res = res.replace(f'$RES{e_i + 1}', '\n        '.join(env_res))
res = res.replace('%', '\%')
res = res.replace('#', '\#')
with open('indexing-results.tex', 'w') as outf:
    outf.write(res)


#####################################
##### TABLE WITH HARDWARE ENVS ######
#####################################    
TEX_TABLE_HARDWARE_ENVS = r'''\resizebox{\linewidth}{!}{
    \begin{tabular}{l||c|c|c|c|c|c}
        \toprule 
        \multirow{2}{*}{Environment Name} & \multirow{2}{*}{CPU Model} & \multirow{2}{*}{GPU Model} & \multirow{2}{*}{Libraries} & \multirow{2}{*}{Versions} & \multicolumn{2}{c}{\# Experiments} \\
        & & & & & Inference & Training \\
        \midrule
        $ENVS
        \endrule
        
    \end{tabular}
}'''

ENV_ROW = r'''\multirow{$X}{*}{$NAME} & \multirow{$X}{*}{$CPU} & \multirow{$X}{*}{$GPU} & $LIBS \\'''
important_libs = ['torch', 'torchvision', 'tensorflow', 'larq', 'onnxruntime', 'onnx']
ENV_NAMES = {
    'A100 x8 - PyTorch 1.10.2+cu113': 'A100 x8 PyTorch',
    'A100 x8 - TensorFlow 2.8.0': 'A100 x8 TensorFlow',
    'RTX 5000 - PyTorch 1.10.2+cu113': 'RTX 5000 PyTorch',
    'RTX 5000 - TensorFlow 2.4.1': 'RTX 5000 TensorFlow',
    'Xeon(R) W-2155 - PyTorch 1.10.2+cu113': 'Xeon W-2155 PyTorch',
    'Xeon(R) W-2155 - TensorFlow 2.4.1': 'Xeon W-2155 PyTorch',
    'Xeon(R) W-2155 - ONNX PT Export 1.10.0': 'Xeon W-2155 ONNX PT',
    'Xeon(R) W-2155 - ONNX TF Export 1.10.0': 'Xeon W-2155 ONNX TF',
}
envs = []
for summary_name, env_name in ENV_NAMES.items():
    no_infer = str(len(LOGS['inference'][summary_name]))
    if summary_name in LOGS['training']:
        no_train = str(len(LOGS['training'][summary_name]))
    else:
        no_train = '0'
    vals = LOGS['inference'][summary_name][0]
    exec_ = vals['execution_platform']
    req_ = vals['requirements']
    row = ENV_ROW.replace('$NAME', env_name)
    row = row.replace('$CPU', ' '.join(exec_['Processor'].split()[:3]).replace('(R)', ''))
    if len(exec_['GPU']) > 0:
        gpu_name = exec_['GPU']['0']['Name'].replace('-40GB', '')
        if len(exec_['GPU']) > 1:
            row = row.replace('$GPU', str(len(exec_['GPU'])) + r' $\times$ $GPU')
        row = row.replace('$GPU', gpu_name)
    else:
        row = row.replace('$GPU', 'n.a.')
    backend = load_backend_info(vals['config']['backend'])
    libs = []
    for package in backend["Packages"]:
        if package in important_libs:
            for req in vals['requirements']:
                if req.split('==')[0].replace('-', '_') == package.replace('-', '_'):
                    lib_str = package + ' & ' + req.split('==')[1]
                    if len(libs) == 0:
                        lib_str += r' & \multirow{$X}{*}{$I} & \multirow{$X}{*}{$T}'.replace('$I', no_infer).replace('$T', no_train)
                    else:
                        lib_str += '& & '
                    libs.append(lib_str)
                    break
    row = row.replace('$LIBS', r''' \\ & & & '''.join(libs))
    row = row.replace('$X', str(len(libs)))
    envs.append(row)
table = TEX_TABLE_HARDWARE_ENVS.replace('$ENVS', '\n        \midrule\n        '.join(envs))
with open('exec-envs.tex', 'w') as outf:
    outf.write(table)


##########################################
##### BAR PLOT WITH RATINGS PER ENV ######
##########################################

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

fig, ax = plt.subplots(1, 2, sharey=True)
fig.set_figwidth(9)
fig.set_figheight(3.5)
for ai, rating_mode in enumerate([RATING_MODE, 'pessimistic mean']):
    ys = np.zeros((5, len(SUMMARIES['inference'])))
    env_labels = sorted(list(SUMMARIES['inference'].keys()))
    for e_i, env in enumerate(env_labels):
        for sum in SUMMARIES['inference'][env]:
            rating = calculate_compound_rating(sum, rating_mode)
            ys[rating, e_i] += 1
    nr_ticks, bars_per_tick = ys.shape
    x_positions = np.arange(nr_ticks)
    for ii, (x, y) in enumerate(zip(x_positions, ys)):
        nr_bars = len(y)
        width = calculate_bar_width(nr_bars, desired_padding=0.02)
        xs = calculate_x_positions(nr_bars, width) + x
        for i, (_x, _y) in enumerate(zip(xs,y)):
            if ii > 0 or ai > 0:
                ax[ai].bar(x=_x, height=_y, width=width, color=COLORS[ii], hatch=HATCHES[i], edgecolor='black', alpha=.8)
            else:
                ax[ai].bar(x=_x, height=_y, width=width, color=COLORS[ii], hatch=HATCHES[i], edgecolor='black', alpha=.8, label=env_labels[i])
    ax[ai].set_xlabel(f'Compound rating is {rating_mode}')
    ax[ai].set_xticks(np.arange(5))
    ax[ai].set_xticklabels([f"{r}" for r in RATINGS])
    lgd = fig.legend(ncol=2, bbox_to_anchor=(0.5, .88), loc='lower center')
    plt.savefig(f'hist_ratings.pdf', bbox_extra_artists=(lgd, ), bbox_inches='tight')


##########################################
##### SCATTER PLOT POWER VS ACCURACY #####
##########################################

def scatter_models(xmetric, ymetric, xlabel, ylabel, fname, scales=SCALES, ind_or_val='index', xlim=None, ylim=None, named_pos=None, named_pos_discard=True, envs=THREE_ENVS):
    fig, ax = plt.subplots(1,1)
    for xi, (x0, x1) in enumerate(scales[xmetric]):
            for yi, (y0, y1) in enumerate(scales[ymetric]):
                x0 = min(50, x0)
                x1 = min(50, x1)
                y0 = min(50, y0)
                y1 = min(50, y1)
                color = calculate_compound_rating([xi, yi], RATING_MODE, COLORS)
                ax.add_patch(Rectangle((x1, y1), x0-x1, y0-y1, color=color, alpha=.6, zorder=-1))
    xmin, xmax, ymin, ymax = 1e12, 0, 1e12, 0
    plt.axhline(y=1, color='w', linestyle='--')
    plt.axvline(x=1, color='w', linestyle='--')
    for e_i, env in enumerate(envs):
        x, y, n, r = [], [], [], []
        for model_sum in SUMMARIES['inference'][env]:
            x.append(model_sum[xmetric][ind_or_val])
            y.append(model_sum[ymetric][ind_or_val])
            n.append(model_sum['name'])
            r.append(calculate_compound_rating(model_sum, RATING_MODE, COLORS))
        xmin = min(xmin, min(x))
        xmax = max(xmax, max(x))
        ymin = min(ymin, min(y))
        ymax = max(ymax, max(y))
        ax.scatter(x, y, s=75, label=env, marker=MARKERS[e_i], color=r, edgecolors='white')
        if e_i == 0:
            for i, name in enumerate(n):
                if named_pos is None or (named_pos_discard and name not in named_pos) or (not named_pos_discard and name in named_pos):
                    ax.annotate(name, (x[i], y[i]))
    if xlim is None:
        xlim = [xmin - (xmax-xmin) * 0.05, xmax + (xmax-xmin) * 0.05]
    if ylim is None:
        ylim = [ymin - (ymax-ymin) * 0.05, ymax + (ymax-ymin) * 0.05]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    leg = fig.legend()
    for lh in leg.legendHandles:
        lh.set_color(RA)
    plt.tight_layout()
    plt.savefig(fname)

not_name = ['EfficientNetB5', 'EfficientNetB6', 'DenseNet201', 'DenseNet169', 'ResNet152', 'InceptionResNetV2', 'NASNetMobile', 'VGG16']

scatter_models('inference_power_draw', 'top1_val',
    'Inference Power Draw / Sample (Index Scale)',
    'Top-1 Validation Accuracy (Index Scale)',
    'scatter_power_acc.pdf', xlim=(0, 2.3), ylim=(0.76, 1.15), named_pos=not_name)


##############################################
##### SCATTER PLOT PARAMETERS VS RUNTIME #####
##############################################

not_name = ['DenseNet121', 'QuickNetLarge', 'InceptionResNetV2', 'InceptionV3']

scatter_models('parameters', 'inference_time',
    'Number of Parameters (Index Scale)',
    'Runtime per Sample (Index Scale)',
    'scatter_parameters_runtime.pdf', xlim=(0.1, 12), ylim=(0.25, 1.6), named_pos=not_name)


############################
##### EXEMPLARY LABELS #####
############################

models = ['ResNet101', 'EfficientNetB0', 'VGG19', 'MobileNetV2']
for env in THREE_ENVS:
    for model in models:
        for m_sum in SUMMARIES['inference'][env]:
            if m_sum['name'] == model:
                pdf_doc = EnergyLabel(m_sum, RATING_MODE)
                pdf_doc.save(f'label_{env}_{m_sum["name"]}.pdf')
                break
