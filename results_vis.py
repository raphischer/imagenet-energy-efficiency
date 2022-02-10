import json

import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from plotly import colors
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from mlel.ratings import rate_results, load_scale, aggregate_rating, KEYS


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLOR_SCALE = colors.make_colorscale(['rgb(0,255,0)', 'rgb(255,255,0)', 'rgb(255,0,0))'])
AXIS_NAMES = {
    "parameters": "Number of Parameters",
    "fsize": "Model File Size [B]", 
    "power_draw": "Power Draw / Sample [Ws]",
    "inference_time": "Inference Time / Sample [ms]",
    "top1_val": "Top-1 Validation Accuracy [%]",
    "top5_val": "Top-5 Validation Accuracy [%]"
}


def add_rating_background(fig, rating_pos, r_colors, mode):
    for xi, (x0, x1) in enumerate(rating_pos[0]):
        for yi, (y0, y1) in enumerate(rating_pos[1]):
            color = aggregate_rating([xi, yi], mode, r_colors)
            fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)


def model_results_to_str(model, environment, rating_mode):
    all_ratings = [val['rating'] for val in model.values() if 'rating' in val]
    final_rating = aggregate_rating(all_ratings, rating_mode)
    environment = f"({environment} Environment)"
    ret_str = [f'Name: {model["name"]:17} {environment:<34} - Final Rating {final_rating}']
    for key, val in model.items():
        if key != 'name':
            ret_str.append(f'{AXIS_NAMES[key]:<30}: {val["value"]:<13.3f} - Index {val["index"]:4.2f} - Rating {val["rating"]}')
    full_str = '\n'.join(ret_str)
    # print(full_str)
    return full_str


def create_scatter_fig(scatter_pos, axis_title, names, env_names, r_colors, ax_border=0.1):
    fig = go.Figure()
    posx, posy = scatter_pos
    for env_i, (x, y, env_name, env, rcol) in enumerate(zip(posx, posy, names, env_names, r_colors)):
        fig.add_trace(go.Scatter(
            x=x, y=y, text=env_name, mode='markers', name=env, marker_symbol=ENV_SYMBOLS[env_i],
            marker=dict(color=rcol, size=15), marker_line=dict(width=3, color='black'))
        )
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
    fig.update_layout(legend=dict(x=.1, y=-.2, orientation="h"))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
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
        self.yaxis_default = 'power_draw'
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
                html.H2('Rating mode:'),
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
                        {'label': AXIS_NAMES[env], 'value': env} for env in KEYS
                    ],
                    value=self.xaxis_default
                ),
            ]),
            html.Div(children=[
                html.H2('Y-Axis:'),
                dcc.Dropdown(
                    id='yaxis',
                    options=[
                        {'label': AXIS_NAMES[env], 'value': env} for env in KEYS
                    ],
                    value=self.yaxis_default
                ),
            ]),
        ])
        self.callback(Output('fig', 'figure'), [Input('environments', 'value'), Input('scale-switch', 'value'), Input('rating', 'value'), Input('xaxis', 'value'), Input('yaxis', 'value')]) (self.update_fig)
        self.callback(Output('model-text', 'children'), Input('fig', 'hoverData'), State('environments', 'value'), State('rating', 'value')) (self.display_model)

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
            names.append([r['name'] for r in self.rated_results[env]])
            x.append([r[xaxis]['value'] for r in self.rated_results[env]])
            y.append([r[yaxis]['value'] for r in self.rated_results[env]])
            x_ind.append([r[xaxis]['index'] for r in self.rated_results[env]])
            y_ind.append([r[yaxis]['index'] for r in self.rated_results[env]])            
            rating_cols.append([aggregate_rating([r[xaxis]['rating'], r[yaxis]['rating']], rating_mode, self.rating_colors) for r in self.rated_results[env]])
        xaxis_name, yaxis_name = AXIS_NAMES[xaxis], AXIS_NAMES[yaxis]
        if scale_switch == 'index':
            scatter_pos = [x_ind, y_ind]
            rating_pos = [self.scales[xaxis], self.scales[yaxis]]
            xaxis_name = xaxis_name.split('[')[0].strip() + ' Index'
            yaxis_name = yaxis_name.split('[')[0].strip() + ' Index'
        else:
            scatter_pos = [x, y]
            rating_pos = [self.scales_real[env_names[0]][xaxis], self.scales_real[env_names[0]][yaxis]]
        fig = create_scatter_fig(scatter_pos, [xaxis_name, yaxis_name], names, env_names, rating_cols)
        add_rating_background(fig, rating_pos, self.rating_colors, rating_mode)
        return fig

    def display_model(self, hover_data=None, env_names=None, rating_mode=None):
        if hover_data is None:
            return 'no model info to show'
        if env_names is None:
            env_names = [list(self.rated_results.keys())[0]]
        if rating_mode is None:
            rating_mode = 'mean'
        point = hover_data['points'][0]
        env_name = env_names[point['curveNumber']]
        model = self.rated_results[env_name][point['pointNumber']]
        return model_results_to_str(model, env_name, rating_mode)


if __name__ == '__main__':

    result_files = {
        'A100_Tensorflow': 'results/A100/results_tf_pretrained.json',
        'A100_PyTorch': 'results/A100/results_torch_pretrained.json',
        'RTX5000_Tensorflow': 'results/RTX5000/results_tf_pretrained.json',
        'RTX5000_PyTorch': 'results/RTX5000/results_torch_pretrained.json',
    }

    app = Visualization(result_files)
    app.run_server(debug=True, host='0.0.0.0', port=8888)