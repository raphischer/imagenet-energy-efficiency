import json

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def calc_accuracy(res, train=False, top5=False):
    split = 'train' if train else 'validation'
    metric = 'sparse_top_k_categorical_accuracy' if top5 else 'sparse_categorical_accuracy'
    return res[split]['results']['metrics'][metric]


def calc_parameters(res):
    return res['train']['results']['model']['params']


def calc_inf_time(res):
    return res['train']['duration'] / 1281167 * 1000


def calc_power_draw(res):
    return res['train']["gpu_monitoring"]["total"]["total_power_draw"] / 1281167


def create_scatter_fig(x, y, axis_title, names, spacing):
    fig = go.Figure()
    # areas
    spacing = [0] + list(np.cumsum(spacing))
    n_lines = len(spacing)
    min_x, min_y = min(x), min(y)
    max_x, max_y = max(x), max(y)
    diff_x, diff_y = max_x - min_x, max_y - min_y
    m = (max_y - min_y) / (max_x - min_x)
    # find graphs for lowest right and uppest left point y = m * x + b
    upper_b = -100000
    lower_b = 10000
    for xv, yv in zip(x, y):
        b = yv - xv * m
        if xv * m + b < xv * m + lower_b:
            lower_b = b
        if xv * m + b > xv * m + upper_b:
            upper_b = b
    b_diff = (upper_b - lower_b)
    x1 = min(x) - (max(x) - min(x))
    x2 = max(x) + (max(x) - min(x))
    colors = ['black', 'green', 'yellow', 'orange', 'red']
    for ar in range(n_lines):
        y1 = x1 * m + lower_b + spacing[ar] * b_diff
        y2 = x2 * m + lower_b + spacing[ar] * b_diff
        if ar == 0:
            y1 -= 10000
            y2 -= 10000
        if ar == n_lines - 1:
            y1 += 10000
            y2 += 10000
        fill = None if ar == 0 else 'tonexty'
        fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], fill=fill, fillcolor=colors[ar], mode='lines', showlegend=False, hoverinfo=None))
    # points
    fig.add_trace(go.Scatter(x=x, y=y, text=names, mode='markers'))
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
    fig.update_layout(xaxis_range=[min(x) - .05 * diff_x, max(x) + .05 * diff_x], yaxis_range=[min(y) - .05 * diff_y, max(y) + .05 * diff_y])
    return fig


class Visualization(dash.Dash):

    def __init__(self, results):
        super().__init__(__name__)
        self.environments = results
        self.spacings = {
            '5%, 15%, 35%, 50%': [0.05, 0.15, 0.35, 0.5],
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
                html.H2('Environment:'),
                dcc.Dropdown(
                    id='environment',
                    options=[
                        {'label': env, 'value': env} for env in self.environments.keys()
                    ],
                    value=list(self.environments.keys())[0]
                ),
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
            results = list(self.environments.values())[0]
        else:
            results = self.environments[env]
        if spac is None:
            spacing = list(self.spacings.values())[0]
        else:
            spacing = self.spacings[spac]
        names = list(results.keys())
        func_x = self.axis_options[self.xaxis_default] if xaxis is None else self.axis_options[xaxis]
        func_y = self.axis_options[self.yaxis_default] if yaxis is None else self.axis_options[yaxis]
        x = [func_x(res) for res in results.values()]
        y = [func_y(res) for res in results.values()]
        return create_scatter_fig(x, y, ['Accuracy', 'Power Draw (Wh)'], names, spacing)


if __name__ == '__main__':

    result_files = {
        'TF_A100': 'results_a100.json',
        'TF_RTX5000': 'results_rtx5000.json'
    }
    results = {}
    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            results[name] = json.load(r)

    app = Visualization(results)
    app.run_server(debug=True)