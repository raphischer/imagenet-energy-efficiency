import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from mlel.ratings import HIGHER_BETTER, load_results, KEYS

raw_symbols = SymbolValidator().values
symbols = [raw_symbols[i] for i in range(0, len(raw_symbols), 12)]


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
        self.xaxis_default = 'top1_val'
        self.yaxis_default = 'parameters'
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
            # html.Div(children=[
            #     html.H2('Distribution:'),
            #     dcc.Dropdown(
            #         id='spacing',
            #         options=[
            #             {'label': spac, 'value': spac} for spac in self.spacings.keys()
            #         ],
            #         value=list(self.spacings.keys())[0]
            #     ),
            # ])
        ])
        self.callback(Output('fig', 'figure'), [Input('scale-switch', 'value'), Input('xaxis', 'value'), Input('yaxis', 'value')]) (self.update_fig)

    def update_fig(self, scale_switch=None, xaxis=None, yaxis=None):
        if scale_switch is None:
            scale_switch = 'index'
        env_names = []
        if xaxis is None:
            xaxis = self.xaxis_default
        if yaxis is None:
            yaxis = self.yaxis_default
        x, x_ind, y, y_ind, x_ratings, y_ratings, names = [], [], [], [], [], [], []
        # access real values
        for env in list(self.environments.keys()):
            env_names.append(env)
            x.append([r[xaxis] for r in results[env]])
            y.append([r[yaxis] for r in results[env]])
            names.append([r['name'] for r in results[env]])
            x_ind.append([r['indices'][xaxis]['value'] for r in results[env]])
            y_ind.append([r['indices'][yaxis]['value'] for r in results[env]])
            x_ratings.append([r['indices'][xaxis]['rating'] for r in results[env]])
            y_ratings.append([r['indices'][yaxis]['rating'] for r in results[env]])
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
    results = load_results(result_files)

    app = Visualization(results)
    app.run_server(debug=True)