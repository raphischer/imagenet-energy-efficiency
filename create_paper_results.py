# importing package
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from mlee.ratings import load_results, rate_results, calculate_compound_rating, load_backend_info
from mlee.label_generator import EnergyLabel

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
THREE_ENVS = ['A100 x8 - TensorFlow 2.8.0', 'RTX 5000 - TensorFlow 2.4.1', 'Xeon(R) W-2155 - ONNX TF Export 1.10.0']
ENV_NAMES = {
    'A100 x8 - TensorFlow 2.8.0': 'A100 x8 TensorFlow',
    'A100 x8 - PyTorch 1.10.2+cu113': 'A100 x8 PyTorch',
    'RTX 5000 - TensorFlow 2.4.1': 'RTX 5000 TensorFlow',
    'RTX 5000 - PyTorch 1.10.2+cu113': 'RTX 5000 PyTorch',
    'Xeon(R) W-2155 - TensorFlow 2.4.1': 'Xeon W-2155 TensorFlow',
    'Xeon(R) W-2155 - PyTorch 1.10.2+cu113': 'Xeon W-2155 PyTorch',
    'Xeon(R) W-2155 - ONNX TF Export 1.10.0': 'Xeon W-2155 ONNX TF',
    'Xeon(R) W-2155 - ONNX PT Export 1.10.0': 'Xeon W-2155 ONNX PT',
}

os.chdir('paper_results')


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
        \bottomrule
        
    \end{tabular}
}'''

ENV_ROW = r'''\multirow{$X}{*}{$NAME} & \multirow{$X}{*}{$CPU} & \multirow{$X}{*}{$GPU} & $LIBS \\'''
important_libs = ['torch', 'torchvision', 'tensorflow', 'larq', 'onnxruntime', 'onnx']
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
##### TABLE WITH INDEXING RESULTS ########
##########################################

TEX_TABLE_INDEXING_RESULTS = r'''\resizebox{\linewidth}{!}{
    \begin{tabular}{l|l|c|c|c|c|c|c|c|c}
        \toprule 
        \multirow{2}{*}{Environment} & \multirow{2}{*}{Model Name}  & \multicolumn{2}{c}{$METR1} & \multicolumn{2}{c}{$METR2} & \multicolumn{2}{c}{$METR3} & \multicolumn{2}{c}{$METR4}  \\  \cline{3-10}
        & & $VALIND \\ 
        \midrule
        $RES
        \bottomrule
    \end{tabular}
    }'''
TABLE_NAMES = {
    "inference_power_draw": ("Power $m_{Ps}$", '[Ws]'),
    "inference_time": ("Time $m_{Ts}$", '[ms]'),
    "parameters": ("Size $m_S$", ''),
    "top1_val": ("Quality $m_{Q1}$", '[%]')
}

ENV_ROW1 = r'''\multirow{$X}{*}{$ENV} &'''
            
res = TEX_TABLE_INDEXING_RESULTS
THREE_MODELS = ['ResNet101', 'EfficientNetB0']
for summary_name, env_name in ENV_NAMES.items():
    row_start = ENV_ROW1.replace('$ENV', f'{env_name}')
    row_start = row_start.replace('$X', str(len(THREE_MODELS)))
    env_res = []
    for model in THREE_MODELS:
        model_res = [model]
        for mod_sum in SUMMARIES['inference'][summary_name]:
            if mod_sum['name'] == model:
                val_ind_txt = []
                for m_i, (metr, (name, unit)) in enumerate(TABLE_NAMES.items()):
                    res = res.replace(f'$METR{m_i + 1}', name)
                    val_ind_txt.append(f'Value {unit} & Index')
                    ind = r'\colorbox{R' + RATINGS[mod_sum[metr]['rating']] + r'}{' + f"{mod_sum[metr]['index']:4.2f}" + '}'
                    if metr == 'parameters':
                        val = f"{mod_sum[metr]['value']:4.1f}e6"
                    else:
                        val = f"{mod_sum[metr]['value']:4.3f}"
                    model_res.extend([val, ind])
                res = res.replace('$VALIND', ' & '.join(val_ind_txt))
                break
        env_res.append(' & '.join(model_res) + r' \\')
    res = res.replace('$RES', row_start + '\n        & '.join(env_res) + '\n        \midrule\n        $RES')
res = res.replace('%', '\%')
res = res.replace('#', '\#')
res = res.replace('\midrule\n        $RES', '')
with open('indexing-results.tex', 'w') as outf:
    outf.write(res)


##########################################
##### SCATTER PLOT POWER VS ACCURACY #####
##########################################

def scatter_models(xmetric, ymetric, xlabel, ylabel, fname, task='inference', scales=SCALES, ind_or_val='index', xlim=None, ylim=None, named_pos=None, named_pos_discard=True, envs=['A100 x8 - TensorFlow 2.8.0'], width=None, height=None):
    fig, ax = plt.subplots(1, 1)
    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)
    for xi, (x1, x0) in enumerate(scales[xmetric]):
        if xi == 0:
            x1 = 100 if xlim is None else xlim[1]
        if xi == len(scales[xmetric]) - 1:
            x0 = 0 if xlim is None else xlim[0]
        for yi, (y1, y0) in enumerate(scales[ymetric]):
            if yi == 0:
                y1 = 100 if xlim is None else ylim[1]
            if yi == len(scales[ymetric]) - 1:
                y0 = 0 if xlim is None else ylim[0]
            color = calculate_compound_rating([xi, yi], RATING_MODE, COLORS)
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, color=color, alpha=.6, zorder=-1))
    xmin, xmax, ymin, ymax = 1e12, 0, 1e12, 0
    plt.axhline(y=1, color='w', linestyle='--')
    plt.axvline(x=1, color='w', linestyle='--')
    for e_i, env in enumerate(envs):
        x, y, n, r = [], [], [], []
        for model_sum in SUMMARIES[task][env]:
            x.append(model_sum[xmetric][ind_or_val] or 0)
            y.append(model_sum[ymetric][ind_or_val] or 0)
            n.append(model_sum['name'])
            r.append(calculate_compound_rating(model_sum, RATING_MODE, COLORS))
        xmin = min(xmin, min(x))
        xmax = max(xmax, max(x))
        ymin = min(ymin, min(y))
        ymax = max(ymax, max(y))
        ax.scatter(x, y, s=75, label=ENV_NAMES[env], marker=MARKERS[e_i], color=r, edgecolors='white')
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
    if len(envs) > 1:
        leg = fig.legend()
        for lh in leg.legendHandles:
            lh.set_color(RA)
    plt.tight_layout()
    plt.savefig(fname)

not_name = ['EfficientNetB5', 'EfficientNetB6', 'DenseNet201', 'DenseNet169', 'ResNet152', 'InceptionResNetV2', 'VGG16']

scatter_models('inference_power_draw', 'top1_val',
    'Inference Power Draw per Sample (Index Scale)',
    'Top-1 Validation Accuracy (Index Scale)',
    'scatter_power_acc.pdf', xlim=(0.1, 2.05), ylim=(0.77, 1.15), named_pos=not_name, width=5, height=5)


##############################################
##### SCATTER PLOT PARAMETERS VS RUNTIME #####
##############################################

not_name = ['DenseNet121', 'QuickNetLarge', 'InceptionResNetV2', 'InceptionV3', 'Xception', 'MobileNetV3Large', 'ResNet152']

scatter_models('parameters', 'inference_time',
    'Number of Parameters (Index Scale)',
    'Runtime per Sample (Index Scale)',
    'scatter_parameters_runtime.pdf', xlim=(0.1, 25.5), ylim=(0.29, 1.525), named_pos=not_name, width=5, height=5)


#####################################################
##### SCATTER PLOT TRAIN POWER DRAW VS ACCURACY #####
#####################################################

not_name = ['ResNet152', 'ResNet50', 'DenseNet121', 'EfficientNetB3']

scatter_models('train_power_draw', 'top1_val',
    'Total Power Draw (Index Scale)',
    'Top-1 Validation Accuracy (Index Scale)',
    'scatter_train_power_acc.pdf', task='training', xlim=(-0.03, 1.38), ylim=(0.77, 1.15), named_pos=not_name, width=6, height=4)


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

fig, ax = plt.subplots(1, 1)
fig.set_figwidth(7)
fig.set_figheight(3)
for ai, rating_mode in enumerate([RATING_MODE]):
    ys = np.zeros((len(ENV_NAMES), 5))
    for e_i, (env, env_name) in enumerate(ENV_NAMES.items()):
        for sum in SUMMARIES['inference'][env]:
            rating = calculate_compound_rating(sum, rating_mode)
            ys[e_i, rating] += 1
    nr_ticks, bars_per_tick = ys.shape
    x_positions = np.arange(nr_ticks)
    for ii, (x, y) in enumerate(zip(x_positions, ys)):
        nr_bars = len(y)
        width = calculate_bar_width(nr_bars, desired_padding=0.05)
        xs = calculate_x_positions(nr_bars, width) + x
        for i, (_x, _y) in enumerate(zip(xs, y)):
            if ii > 0 or ai > 0:
                ax.bar(x=_x, height=_y, width=width, color=COLORS[i], edgecolor='black', alpha=.8)
            else:
                ax.bar(x=_x, height=_y, width=width, color=COLORS[i], edgecolor='black', alpha=.8, label=RATINGS[i])
    # ax.set_xlabel(f'Compound rating is {rating_mode}')
    ax.set_xticks(np.arange(len(ENV_NAMES.keys())))
    ax.set_xticklabels(list(ENV_NAMES.values()), rotation=45, ha='right')
    ax.set_ylabel('Model Rating Frequency')
    lgd = fig.legend(bbox_to_anchor=(0.9, 0.88), loc='upper right')
plt.savefig(f'hist_ratings.pdf', bbox_inches='tight')


############################
##### EXEMPLARY LABELS #####
############################

env = 'A100 x8 - TensorFlow 2.8.0'
for m_sum in SUMMARIES['inference'][env]:
    if m_sum['name'] == 'MobileNetV2':
        pdf_doc = EnergyLabel(m_sum, RATING_MODE)
        pdf_doc.save(f'label_{env}_{m_sum["name"]}.pdf')
        break

for m_sum in SUMMARIES['training'][env]:
    if m_sum['name'] == 'EfficientNetB2':
        pdf_doc = EnergyLabel(m_sum, RATING_MODE)
        pdf_doc.save(f'label_{env}_{m_sum["name"]}.pdf')
        break