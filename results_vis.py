import json

import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from plotly import colors
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from mlel.ratings import HIGHER_BETTER, rate_results, load_scale, aggregate_rating, KEYS


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLOR_SCALE = colors.make_colorscale(['rgb(0,255,0)', 'rgb(255,255,0)', 'rgb(255,0,0))'])


def add_rating_background(fig, rating_pos, r_colors, mode):
    for xi, (x0, x1) in enumerate(rating_pos[0]):
        for yi, (y0, y1) in enumerate(rating_pos[0]):
            color = aggregate_rating([xi, yi], mode, r_colors)
            fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)


def create_scatter_fig(scatter_pos, axis_title, names, env_names, r_colors, ax_border=0.1):
    fig = go.Figure()
    # areas
    if axis_title[0] in HIGHER_BETTER:
        direction = 'UR' if axis_title[1] in HIGHER_BETTER else 'LR'
    else:
        direction = 'UL' if axis_title[1] in HIGHER_BETTER else 'LL'
    # points
    posx, posy = scatter_pos
    for env_i, (x, y, env_name, env, rcol) in enumerate(zip(posx, posy, names, env_names, r_colors)):
        fig.add_trace(go.Scatter(
            x=x, y=y, text=env_name, mode='markers', name=env, marker_symbol=ENV_SYMBOLS[env_i],
            marker=dict(color=rcol, size=15), marker_line=dict(width=3, color='white'))
        )
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1], title=direction)
    fig.update_layout(legend=dict(x=.1, y=-.2, orientation="h"))
    # fig.update_layout(clickmode='event')
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
        self.results = results
        self.xaxis_default = 'top1_val'
        self.yaxis_default = 'parameters'
        self.reference_name = 'ResNet101'
        self.scales = load_scale()
        self.rating_colors = colors.sample_colorscale(RATING_COLOR_SCALE, samplepoints=[float(c) / (len(list((self.scales.values()))[0]) - 1) for c in range(len(list((self.scales.values()))[0]))])
        self.rated_results, self.scales_real = rate_results(self.results, self.scales, self.reference_name)
        self.layout = html.Div(children=[
            dcc.Graph(
                id='fig',
                figure=self.update_fig(),
                # responsive=True,
                # config={'responsive': True},
                # style={'height': '100%', 'width': '100%'}
            ),
            html.Div(id='model-text', style={'whiteSpace': 'pre-line'}),
            html.Div(children=[
                html.H2('Environments:'),
                dcc.Checklist(
                    id='environments',
                    options=[{'label': env, 'value': env} for env in self.rated_results.keys()],
                    value=[list(self.rated_results.keys())[0]]
                )
            ]),
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
                html.H2('Rating visualization:'),
                dcc.RadioItems(
                    id='rating',
                    options=[{'label': opt, 'value': opt.lower()} for opt in ['Mean', 'Best', 'Worst', 'Majority', 'Median']],
                    value='mean'
                )
            ]),
            html.Div(children=[
                html.H2('X-Axis:'),
                dcc.Dropdown(
                    id='xaxis',
                    options=[
                        {'label': env, 'value': env} for env in KEYS
                    ],
                    value=self.xaxis_default
                ),
            ]),
            html.Div(children=[
                html.H2('Y-Axis:'),
                dcc.Dropdown(
                    id='yaxis',
                    options=[
                        {'label': env, 'value': env} for env in KEYS
                    ],
                    value=self.yaxis_default
                ),
            ]),
        ])
        self.callback(Output('fig', 'figure'), [Input('environments', 'value'), Input('scale-switch', 'value'), Input('rating', 'value'), Input('xaxis', 'value'), Input('yaxis', 'value')]) (self.update_fig)
        self.callback(Output('model-text', 'children'), Input('fig', 'hoverData'), State('environments', 'value')) (self.display_model)

    def update_fig(self, env_names=None, scale_switch=None, rating_mode=None, xaxis=None, yaxis=None):
        if env_names is None:
            env_names = [list(self.rated_results.keys())[0]]
        if scale_switch is None:
            scale_switch = 'index'
        if rating_mode is None:
            rating_mode = 'mean'
        if xaxis is None:
            xaxis = self.xaxis_default
        if yaxis is None:
            yaxis = self.yaxis_default
        x, x_ind, y, y_ind, rating_cols, names = [], [], [], [], [], [],
        # access real values
        for env in env_names:
            x.append([r[xaxis] for r in self.rated_results[env]])
            y.append([r[yaxis] for r in self.rated_results[env]])
            names.append([r['name'] for r in self.rated_results[env]])
            x_ind.append([r[xaxis]['index'] for r in self.rated_results[env]])
            y_ind.append([r[yaxis]['index'] for r in self.rated_results[env]])            
            rating_cols.append([aggregate_rating([r[xaxis]['rating'], r[yaxis]['rating']], rating_mode, self.rating_colors) for r in self.rated_results[env]])
        if scale_switch == 'index':
            scatter_pos = [x_ind, y_ind]
            rating_pos = [self.scales[xaxis], self.scales[yaxis]]
            xaxis = xaxis.split('[')[0].strip() + ' Index'
            yaxis = yaxis.split('[')[0].strip() + ' Index'
        else:
            scatter_pos = [x, y]
            rating_pos = [self.scales_real[env_names[0]][xaxis], self.scales_real[env_names[0]][yaxis]]
        fig = create_scatter_fig(scatter_pos, [xaxis, yaxis], names, env_names, rating_cols)
        add_rating_background(fig, rating_pos, self.rating_colors, rating_mode)
        return fig

    def display_model(self, hover_data=None, env_names=None):
        if hover_data is None:
            return 'no model info to show'
        if env_names is None:
            env_names = [list(self.rated_results.keys())[0]]
        point = hover_data['points'][0]
        model = self.rated_results[env_names[point['curveNumber']]][point['pointNumber']]
        return json.dumps(model, indent=4)


if __name__ == '__main__':

    result_files = {
        'A100_Tensorflow': 'results/A100/results_tf_pretrained.json',
        'A100_PyTorch': 'results/A100/results_torch_pretrained.json',
        'RTX5000_Tensorflow': 'results/RTX5000/results_tf_pretrained.json',
        'RTX5000_PyTorch': 'results/RTX5000/results_torch_pretrained.json',
    }

    app = Visualization(result_files)
    app.run_server(debug=True, host='0.0.0.0', port=8888)