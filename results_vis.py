import json

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator


raw_symbols = SymbolValidator().values
symbols = [raw_symbols[i] for i in range(0, len(raw_symbols), 12)]


KEYS = ["parameters", "fsize", "power_draw", "inference_time", "top1_val", "top5_val"]


RATINGS = ['green', 'yellow', 'orange', 'red', 'gray']


HIGHER_BETTER = [
    'top1_val',
    'top5_val',
]


def load_scale(path="scales.json"):
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


def calculate_index(values, ref, axis):
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


def calc_rating(x, y, bins, direction):
    for b_i, bin in enumerate(bins):
        p = bin(x)
        if y <= p:
            break
    else:
        b_i = 3
    if direction in ['UR', 'UL']:
        return 3 - b_i
    return b_i


# def calculate_rating(x, y, direction='LR', spacing=[0.05, 0.15, 0.35, 0.5], fig=None):
#     fillcols = ['green'] + RATINGS + ['red']
#     if direction in ['UR', 'UL']:
#         spacing = [0] + list(np.cumsum(list(reversed(spacing))))
#         fillcols = list(reversed(fillcols))
#     else: # UR LR
#         spacing = [0] + list(np.cumsum(spacing))
#     n_lines = len(spacing)
#     xf = [v2 for v in x for v2 in v]
#     yf = [v2 for v in y for v2 in v]
#     # calculate the rating lines' slope
#     min_x, max_x, min_y, max_y = min(xf), max(xf), min(yf), max(yf)
#     m = 1 # float(max_y - min_y) / float(max_x - min_x)
#     if direction in ['UR', 'LL']:
#         m = -1.0 / m
#     # find b shift of rating lines for lowest and uppest point y = m * x + b
#     upper_b = -1e21
#     lower_b = 1e21
#     for xv, yv in zip(xf, yf):
#         b = yv - xv * m
#         if xv * m + b < xv * m + lower_b:
#             lower_b = b
#         if xv * m + b > xv * m + upper_b:
#             upper_b = b
#     b_diff = (upper_b - lower_b)
#     # place line start and end outside of the zoomed window
#     x1 = min_x - (max_x - min_x)
#     x2 = max_x + (max_x - min_x)
#     bins = []
#     if fig is not None:
#         # identify individual lines
#         for l in range(n_lines):
#             y1 = x1 * m + lower_b + spacing[l] * b_diff
#             y2 = x2 * m + lower_b + spacing[l] * b_diff
#             bins.append(lambda xin, l=l: xin * m + lower_b + spacing[l] * b_diff)
#             # TODO if uppest or lowest line, shift further up or down
#             # if l == 0:
#             #     y1 -= 10000
#             #     y2 -= 10000
#             # if l == n_lines - 1:
#             #     y1 += 10000
#             #     y2 += 10000
            
#     bins = bins[1:-1]
#     r_func = lambda xin, yin: calc_rating(xin, yin, bins, direction)
#     ratings = []
#     for xv, yv in zip(x, y):
#         ratings.append([r_func(xvv, yvv) for xvv, yvv in zip(xv, yv)])
#     return bins, r_func, ratings


def create_scatter_fig(scatter_pos, rating_pos, axis_title, ratings, names, env_names, ax_border=0.1):
    fig = go.Figure()
    # areas
    if axis_title[0] in HIGHER_BETTER:
        direction = 'UR' if axis_title[1] in HIGHER_BETTER else 'LR'
    else:
        direction = 'UL' if axis_title[1] in HIGHER_BETTER else 'LL'
    # points
    posx, posy = scatter_pos
    for env_i, (x, y, env_name, env) in enumerate(zip(posx, posy, names, env_names)):
        fig.add_trace(go.Scatter(
            x=x, y=y, text=env_name, mode='markers', name=env, marker_symbol=symbols[env_i],
            marker=dict(color='red', size=15), marker_line=dict(width=3, color='white'))
        )
    # areas
    # posx, posy = rating_pos
    # for l, x in enumerate(posx):
    #     fill = None if l == 0 else 'tonexty'
    #     fig.add_trace(go.Scatter(x=x, y=posy[l], fill=fill, mode='lines', line={'color': l}, showlegend=False, hoverinfo=None))
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1], title=direction)
    fig.update_layout(legend=dict(x=.1, y=-.2, orientation="h"))
    min_x, max_x = np.min([min(v) for v in scatter_pos[0]]), np.max([max(v) for v in scatter_pos[0]])
    min_y, max_y = np.min([min(v) for v in scatter_pos[1]]), np.max([max(v) for v in scatter_pos[1]])
    diff_x, diff_y = max_x - min_x, max_y - min_y
    fig.update_layout(
        xaxis_range=[min_x - ax_border * diff_x, max_x + ax_border * diff_x],
        yaxis_range=[min_y - ax_border * diff_y, max_y + ax_border * diff_y]
    )
    return fig


class Visualization(dash.Dash):

    def __init__(self, results):
        super().__init__(__name__)
        self.environments = results
        self.axis_options = {
            'top1_val': lambda res: calc_accuracy(res),
            'top5_val': lambda res: calc_accuracy(res, top5=True),
            'parameters': calc_parameters,
            'fsize': calc_fsize,
            'inference_time': calc_inf_time,
            'power_draw': calc_power_draw
        }
        self.xaxis_default = 'top1_val'
        self.yaxis_default = 'parameters'
        self.scales = load_scale()
        self.reference_name = 'ResNet101'
        self.layout = html.Div(children=[
            dcc.Graph(
                id='fig',
                figure=self.update_fig(),
                # responsive=True,
                # config={'responsive': True},
                # style={'height': '100%', 'width': '100%'}
            ),
            html.Div(children=[
                html.H2('Axis Scales:'),
                dcc.RadioItems(
                    id='scale-switch',
                    options=[
                        {'label': 'Reference Index', 'value': 'index'},
                        {'label': 'Real Values', 'value': 'real'}
                    ],
                    value='index'
                )
            ]),
            html.Div(children=[
                html.H2('X-Axis:'),
                dcc.Dropdown(
                    id='xaxis',
                    options=[
                        {'label': env, 'value': env} for env in self.axis_options.keys()
                    ],
                    value=self.xaxis_default
                ),
            ]),
            html.Div(children=[
                html.H2('Y-Axis:'),
                dcc.Dropdown(
                    id='yaxis',
                    options=[
                        {'label': env, 'value': env} for env in self.axis_options.keys()
                    ],
                    value=self.yaxis_default
                ),
            ]),
            html.Div(children=[
                html.H2('Distribution:'),
                dcc.Dropdown(
                    id='spacing',
                    options=[
                        {'label': spac, 'value': spac} for spac in self.spacings.keys()
                    ],
                    value=list(self.spacings.keys())[0]
                ),
            ])
        ])
        self.callback(Output('fig', 'figure'), [Input('scale-switch', 'value'), Input('xaxis', 'value'), Input('yaxis', 'value')]) (self.update_fig)

    def update_fig(self, scale_switch=None, xaxis=None, yaxis=None):
        if scale_switch is None:
            scale_switch = 'index'
        env_names = list(self.environments.keys())
        results = [self.environments[e].values() for e in env_names]
        if xaxis is None:
            xaxis = self.xaxis_default
        if yaxis is None:
            yaxis = self.yaxis_default
        func_x, func_y = self.axis_options[xaxis], self.axis_options[yaxis]
        x, x_ind, y, y_ind, x_ratings, y_ratings, names = [], [], [], [], [], [], []
        # access real values
        for result in results:
            x.append([float(func_x(res)) for res in result])
            y.append([float(func_y(res)) for res in result])
            names.append([res['config']['eval_model'] for res in result])
        # calculate index values & rankings
        for i, result in enumerate(results):
            idx = names[i].index(self.reference_name)
            x_ind.append(calculate_index(x[i], x[i][idx], xaxis))
            y_ind.append(calculate_index(y[i], y[i][idx], yaxis))
            x_ratings.append(calculate_rating(x_ind[-1], self.scales[xaxis]))
            y_ratings.append(calculate_rating(y_ind[-1], self.scales[yaxis]))
        if scale_switch == 'index':
            xaxis = xaxis.split('[')[0].strip() + ' Index'
            yaxis = yaxis.split('[')[0].strip() + ' Index'
            scatter_pos = [x_ind, y_ind]
            # rating_pos = [x_ind_ranks, y_ind_ranks]
        else:
            scatter_pos = [x, y]
            # rating_pos = [x_ranks, x_ranks]
        return create_scatter_fig(scatter_pos, [], [xaxis, yaxis], [x_ratings, y_ratings], names, env_names)


if __name__ == '__main__':

    result_files = {
        'A100_Tensorflow': 'results/A100/results_tf_pretrained.json',
        'A100_PyTorch': 'results/A100/results_torch_pretrained.json',
        'RTX5000_Tensorflow': 'results/RTX5000/results_tf_pretrained.json',
        'RTX5000_PyTorch': 'results/RTX5000/results_torch_pretrained.json',
    }
    results = {}
    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            results[name] = json.load(r)

    app = Visualization(results)
    app.run_server(debug=True)