import json

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def create_fig(x, y, axis_title, names, spacing):
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
        self.results = results
        self.spacings = {
            '10%, 20%, 30%, 40%': [0.1, 0.2, 0.3, 0.4],
            '25%, 25%, 25%, 25%': [0.25, 0.25, 0.25, 0.25],
            '5%, 15%, 35%, 50%': [0.05, 0.15, 0.35, 0.5],
        }
        self.layout = html.Div(children=[
            dcc.Graph(
                id='fig',
                figure=self.update_fig(),
                # responsive=True,
                # config={'responsive': True},
                # style={'height': '100%', 'width': '100%'}
            ),
            dcc.Dropdown(
                id='spacing',
                options=[
                    {'label': spac, 'value': spac} for spac in self.spacings.keys()
                ],
                value=list(self.spacings.keys())[0]
            ),
        ])
        self.callback(Output('fig', 'figure'), Input('spacing', 'value')) (self.update_fig)

    def update_fig(self, spac=None):
        if spac is None:
            spacing = list(self.spacings.values())[0]
        else:
            spacing = self.spacings[spac]
        results = self.results['TF_A100']
        names = list(results.keys())
        accuracy = [res['validation']['results']['metrics']['accuracy'] for res in results.values()]
        power_draw = [res['validation']["gpu_monitoring"]["total"]["total_power_draw"] / 3600 for res in results.values()]
        return create_fig(accuracy, power_draw, ['Accuracy', 'Power Draw (Wh)'], names, spacing)


if __name__ == '__main__':

    result_files = {'TF_A100': 'results_a100.json'}
    results = {}
    for name, resf in result_files.items():
        with open(resf, 'r') as r:
            results[name] = json.load(r)

    app = Visualization(results)
    app.run_server(debug=True)