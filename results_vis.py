import json

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go


RATINGS = ['green', 'yellow', 'orange', 'red']


HIGHER_BETTER = [
    'Top-1 Validation Accuracy [%]',
    'Top-1 Training Accuracy [%]',
    'Top-5 Validation Accuracy [%]',
    'Top-5 Training Accuracy [%]'
]


def calc_accuracy(res, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'top_5_accuracy' if top5 else 'accuracy'
    return res[split]['results']['metrics'][metric]


def calc_parameters(res):
    return res['train']['results']['model']['params']


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


def calculate_rating_bins(x, y, direction='LR', spacing=[0.05, 0.15, 0.35, 0.5], fig=None):
    fillcols = ['black'] + RATINGS + ['white']
    if direction in ['UR', 'UL']:
        spacing = [0] + list(np.cumsum(list(reversed(spacing))))
        fillcols = list(reversed(fillcols))
    else: # UR LR
        spacing = [0] + list(np.cumsum(spacing))
    n_lines = len(spacing)
    xf = [v2 for v in x for v2 in v]
    yf = [v2 for v in y for v2 in v]
    # calculate the rating lines' slope
    min_x, max_x, min_y, max_y = min(xf), max(xf), min(yf), max(yf)
    m = (max_y - min_y) / (max_x - min_x)
    if direction in ['UR', 'LL']:
        m *= -1
    # find b shift of rating lines for lowest and uppest point y = m * x + b
    upper_b = -1e21
    lower_b = 1e21
    for xv, yv in zip(xf, yf):
        b = yv - xv * m
        if xv * m + b < xv * m + lower_b:
            lower_b = b
        if xv * m + b > xv * m + upper_b:
            upper_b = b
    b_diff = (upper_b - lower_b)
    # place line start and end outside of the zoomed window
    x1 = min_x - (max_x - min_x)
    x2 = max_x + (max_x - min_x)
    bins = []
    if fig is not None:
        # identify individual lines
        for l in range(n_lines):
            y1 = x1 * m + lower_b + spacing[l] * b_diff
            y2 = x2 * m + lower_b + spacing[l] * b_diff
            bins.append(lambda xin, l=l: xin * m + lower_b + spacing[l] * b_diff)
            # TODO if uppest or lowest line, shift further up or down
            # if l == 0:
            #     y1 -= 10000
            #     y2 -= 10000
            # if l == n_lines - 1:
            #     y1 += 10000
            #     y2 += 10000
            fill = None if l == 0 else 'tonexty'
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], fill=fill, fillcolor=fillcols[l], mode='lines', line={'color': fillcols[l]}, showlegend=False, hoverinfo=None))
    bins = bins[1:-1]
    r_func = lambda xin, yin: calc_rating(xin, yin, bins, direction)
    ratings = []
    for xv, yv in zip(x, y):
        ratings.append([r_func(xvv, yvv) for xvv, yvv in zip(xv, yv)])
    return bins, r_func, ratings


def create_scatter_fig(x, y, axis_title, names, envs, spacing, ax_border=0.1):
    fig = go.Figure()
    # areas
    if axis_title[0] in HIGHER_BETTER:
        direction = 'UR' if axis_title[1] in HIGHER_BETTER else 'LR'
    else:
        direction = 'UL' if axis_title[1] in HIGHER_BETTER else 'LL'
    bins, r_func, ratings = calculate_rating_bins(x, y, direction, spacing, fig)
    # points
    for env_x, env_v, env_names, env, env_rating in zip(x, y, names, envs, ratings):
        # TODO different line colors for the different environments
        fig.add_trace(go.Scatter(
            x=env_x, y=env_v, text=env_names, mode='markers', name=env,
            marker=dict(color=[RATINGS[r] for r in env_rating], size=10), marker_line=dict(width=4, color='white'))
        )
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1], title=direction)
    min_x, max_x = np.min([min(v) for v in x]), np.max([max(v) for v in x])
    min_y, max_y = np.min([min(v) for v in y]), np.max([max(v) for v in y])
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
        self.spacings = {
            '5%, 15%, 30%, 50%': [0.05, 0.15, 0.30, 0.5],
            '10%, 20%, 30%, 40%': [0.1, 0.2, 0.3, 0.4],
            '25%, 25%, 25%, 25%': [0.25, 0.25, 0.25, 0.25],
        }
        self.axis_options = {
            'Top-1 Validation Accuracy [%]': lambda res: calc_accuracy(res),
            'Top-1 Training Accuracy [%]': lambda res: calc_accuracy(res, train=True),
            'Top-5 Validation Accuracy [%]': lambda res: calc_accuracy(res, top5=True),
            'Top-5 Training Accuracy [%]': lambda res: calc_accuracy(res, train=True, top5=True),
            'Parameters': calc_parameters,
            'Inference time / Sample [ms]': calc_inf_time,
            'Power Draw / Sample [Ws]': calc_power_draw
        }
        self.xaxis_default = 'Top-1 Validation Accuracy [%]'
        self.yaxis_default = 'Parameters'
        self.layout = html.Div(children=[
            dcc.Graph(
                id='fig',
                figure=self.update_fig(),
                # responsive=True,
                # config={'responsive': True},
                # style={'height': '100%', 'width': '100%'}
            ),
            html.Div(children=[
                html.H2('Environments:'),
                dcc.Checklist(
                    id='environment',
                    options=[
                        {'label': env, 'value': env} for env in self.environments.keys()
                    ],
                    value=[list(self.environments.keys())[0]]
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
        self.callback(Output('fig', 'figure'), [Input('environment', 'value'), Input('xaxis', 'value'), Input('yaxis', 'value'), Input('spacing', 'value')]) (self.update_fig)

    def update_fig(self, env=None, xaxis=None, yaxis=None, spac=None):
        if env is None:
            env = [list(self.environments.keys())[0]]
        results = [self.environments[e].values() for e in env]
        if spac is None:
            spacing = list(self.spacings.values())[0]
        else:
            spacing = self.spacings[spac]
        if xaxis is None:
            xaxis = self.xaxis_default
        if yaxis is None:
            yaxis = self.yaxis_default
        func_x, func_y = self.axis_options[xaxis], self.axis_options[yaxis]
        x, y, names = [], [], []
        for result in results:
            x.append([func_x(res) for res in result])
            y.append([func_y(res) for res in result])
            names.append([res['config']['eval_model'] for res in result])
        return create_scatter_fig(x, y, [xaxis, yaxis], names, env, spacing)


if __name__ == '__main__':

    result_files = {
        'A100_Tensorflow': 'results/A100/results_tf_pretrained.json',
        'A100_PyTorch': 'results/A100/results_torch_pretrained.json',
        'RTX5000_Tensorflow': 'results/A100/results_tf_pretrained.json',
        'RTX5000_PyTorch': 'results/A100/results_torch_pretrained.json',
    }
    results = {}
    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            results[name] = json.load(r)

    app = Visualization(results)
    app.run_server(debug=True)