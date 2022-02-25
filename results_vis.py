import argparse
import base64
import json

import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from plotly import colors
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from mlel.ratings import load_results, load_scale, save_scale, rate_results, aggregate_rating, TASK_TYPES
from mlel.label_generator import EnergyLabel


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLOR_SCALE = colors.make_colorscale(['rgb(0,255,0)', 'rgb(255,255,0)', 'rgb(255,0,0))'])
RATING_COLORS = colors.sample_colorscale(RATING_COLOR_SCALE, samplepoints=[float(c) / (4) for c in range(5)])
AXIS_NAMES = {
    "parameters": "Number of Parameters",
    "fsize": "Model File Size [B]", 
    "inference_power_draw": "Inference Power Draw / Sample [Ws]",
    "inference_time": "Inference Time / Sample [ms]",
    "train_power_draw": "Full Training Power Draw [Ws]",
    "train_power_draw_epoch": "Training Power Draw per Epoch [Ws]",
    "train_time": "Full Training Time [s]",
    "train_time_epoch": "Training Time per Epoch [s]",
    "top1_val": "Top-1 Validation Accuracy [%]",
    "top5_val": "Top-5 Validation Accuracy [%]"
}
PATTERNS = ["", "/", ".", "x", "-", "\\", "|", "+", "."]


def add_rating_background(fig, rating_pos, mode):
    for xi, (x0, x1) in enumerate(rating_pos[0]):
        for yi, (y0, y1) in enumerate(rating_pos[1]):
            color = aggregate_rating([xi, yi], mode, RATING_COLORS)
            fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8, row=1, col=1)


def summary_to_str(summary, rating_mode):
    final_rating = aggregate_rating(summary, rating_mode)
    environment = f"({summary['environment']} Environment)"
    ret_str = [f'Name: {summary["name"]:17} {environment:<34} - Final Rating {final_rating}']
    for key, val in summary.items():
        if isinstance(val, dict) and "value" in val :
            if val["value"] is None:
                ret_str.append(f'{AXIS_NAMES[key]:<30}: {"n.a.":<13} - Index n.a. - Rating {val["rating"]}')
            else:
                ret_str.append(f'{AXIS_NAMES[key]:<30}: {val["value"]:<13.3f} - Index {val["index"]:4.2f} - Rating {val["rating"]}')
    full_str = '\n'.join(ret_str)
    return full_str


def create_scatter_fig(plot_data, axis_title, ax_border=0.1):
    fig = make_subplots(rows=1, cols=2)
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], text=data['names'], name=env_name, 
            mode='markers', marker_symbol=ENV_SYMBOLS[env_i],
            legendgroup=env_name, marker=dict(color=[RATING_COLORS[r] for r in data['ratings']], size=15),
            marker_line=dict(width=3, color='black')), row=1, col=1
        )
        counts = np.bincount(data['ratings'])
        fig.add_trace(go.Bar(
            name=env_name, x=['A', 'B', 'C', 'D', 'E'], y=counts, legendgroup=env_name,
            marker_pattern_shape=PATTERNS[env_i], marker_color=RATING_COLORS), row=1, col=2)
    fig.update_traces(textposition='top center', row=1, col=1)
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
    # fig.update_layout(legend=dict(x=.1, y=-.2, orientation="h"))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig.update_layout(barmode='stack')
    # fig.update_layout(clickmode='event')
    min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
    min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
    diff_x, diff_y = max_x - min_x, max_y - min_y
    fig.update_layout(
        xaxis_range=[min_x - ax_border * diff_x, max_x + ax_border * diff_x],
        yaxis_range=[min_y - ax_border * diff_y, max_y + ax_border * diff_y]
    )
    return fig


class Visualization(dash.Dash):

    def __init__(self, results_directory, reference_name='ResNet101'):
        super().__init__(__name__)
        self.logs, summaries = load_results(results_directory)
        self.summaries, self.scales, self.scales_real = rate_results(summaries, reference_name)
        self.keys = {task: [key for key, vals in list(self.summaries.values())[0][task][0].items() if isinstance(vals, dict) and 'rating' in vals] for task in TASK_TYPES.values()}
        self.type, self.xaxis, self.yaxis = 'inference', 'top1_val', 'inference_power_draw'
        self.reference_name = reference_name
        self.current = { 'summary': None, 'label': None, 'logs': None }
        # setup page and create callbacks
        self.layout = self.create_page()
        self.callback(
            [Output('xaxis', 'options'), Output('xaxis', 'value'), Output('yaxis', 'options'),  Output('yaxis', 'value')],
            Input('type-switch', 'value')
        ) (self.update_type)
        self.callback(
            [Output(sl_id, prop) for sl_id in ['scaleslider-x', 'scaleslider-y'] for prop in ['min', 'max', 'value', 'marks']],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('scales-upload', 'contents')]
        ) (self.update_sliders)
        self.callback(
            Output('figures', 'figure'),
            [Input('environments', 'value'), Input('scale-switch', 'value'), Input('rating', 'value'), Input('scaleslider-x', 'value'), Input('scaleslider-y', 'value')]
        ) (self.update_figures)
        self.callback(
            [Output('model-text', 'children'), Output('model-label', "src")],
            Input('figures', 'hoverData'), State('environments', 'value'), State('rating', 'value')
        ) (self.display_model)
        self.callback(Output('save-label', 'data'), [Input('btn-save-label', 'n_clicks'), Input('btn-save-summary', 'n_clicks'), Input('btn-save-logs', 'n_clicks')]) (self.save_label)
        self.callback(Output('save-scales', 'data'), Input('btn-save-scales', 'n_clicks')) (self.save_scales)
        
    def create_page(self):
        return html.Div(children=[
            dcc.Graph(
                id='figures',
                figure=self.update_figures(),
                # responsive=True,
                # config={'responsive': True},
                # style={'height': '100%', 'width': '100%'}
            ),
            html.Div(children=[
                html.H2('X-Axis:'),
                dcc.Dropdown(
                    id='xaxis', value=self.xaxis,
                    options=[
                        {'label': AXIS_NAMES[env], 'value': env} for env in self.keys[self.type]
                    ]
                ),
                dcc.RangeSlider(id='scaleslider-x', min=0, max=1, value=[.2, .4, .6, .8], step=.01, pushable=.01, tooltip={"placement": "bottom", "always_visible": True})
            ]),
            html.Div(children=[
                html.H2('Y-Axis:'),
                dcc.Dropdown(
                    id='yaxis', value=self.yaxis,
                    options=[
                        {'label': AXIS_NAMES[env], 'value': env} for env in self.keys[self.type]
                    ]
                ),
                dcc.RangeSlider(id='scaleslider-y', min=0, max=1, value=[.2, .4, .6, .8], step=.01, pushable=.01, tooltip={"placement": "bottom", "always_visible": True})
            ]),
            html.Button("Save Current Scales", id="btn-save-scales"),
            dcc.Download(id="save-scales"),
            dcc.Upload(
                id="scales-upload",
                children=['Drag and Drop or ', html.A('Select a Scales File (.json)')],
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                }
            ),
            html.Div(id='model-text', style={'whiteSpace': 'pre-line'}),
            html.Img(id='model-label', style={"height": "300px"}),
            html.Button("Save Label", id="btn-save-label"),
            html.Button("Save Summary", id="btn-save-summary"),
            html.Button("Save Logs", id="btn-save-logs"),
            dcc.Download(id="save-label"),
            html.Div(children=[
                html.H2('Environments:'),
                dcc.Checklist(
                    id='environments',
                    options=[{'label': env, 'value': env} for env in self.summaries.keys()],
                    value=[list(self.summaries.keys())[0]]
                )
            ]),
            html.Div(children=[
                html.H2('Results Type:'),
                dcc.RadioItems(
                    id='type-switch', value=self.type,
                    options=[{'label': restype.capitalize(), 'value': restype} for restype in self.keys.keys()],
                )
            ]),
            html.Div(children=[
                html.H2('Axis Scales:'),
                dcc.RadioItems(
                    id='scale-switch', value='index',
                    options=[
                        {'label': 'Reference Index', 'value': 'index'},
                        {'label': 'Real Values', 'value': 'real'}
                    ],
                )
            ]),
            html.Div(children=[
                html.H2('Rating mode:'),
                dcc.RadioItems(
                    id='rating', value='optimistic median',
                    options=[{'label': opt, 'value': opt.lower()} for opt in ['Optimistic Median', 'Pessimistic Median', 'Optimistic Mean', 'Pessimistic Mean', 'Best', 'Worst']],
                )
            ])
        ])

    def update_figures(self, env_names=None, scale_switch=None, rating_mode=None, *slider_args):
        if any(slider_args) and 'slider' in dash.callback_context.triggered[0]['prop_id']:
            self.update_scales(slider_args)
        env_names = [list(self.summaries.keys())[0]] if env_names is None else env_names
        scale_switch = 'index' if scale_switch is None else scale_switch
        rating_mode = 'mean' if rating_mode is None else rating_mode
        plot_data = {}
        for env in env_names:
            if len(self.summaries[env][self.type]) > 0:
                env_data = { 'names': [], 'ratings':[], 'x': [], 'y': [] }
                for sum in self.summaries[env][self.type]:
                    env_data['names'].append(sum['name'])
                    env_data['ratings'].append(aggregate_rating(sum, rating_mode))
                    if scale_switch == 'index':
                        env_data['x'].append(sum[self.xaxis]['index'] or 0)
                        env_data['y'].append(sum[self.yaxis]['index'] or 0)
                    else:
                        env_data['x'].append(sum[self.xaxis]['value'] or 0)
                        env_data['y'].append(sum[self.yaxis]['value'] or 0)
                plot_data[env] = env_data
        scale_names = [AXIS_NAMES[self.xaxis], AXIS_NAMES[self.yaxis]]
        if scale_switch == 'index':
            rating_pos = [self.scales[self.xaxis], self.scales[self.yaxis]]
            scale_names = [name.split('[')[0].strip() + ' Index' for name in scale_names]
        else:
            rating_pos = [self.scales_real[env_names[0]][self.xaxis], self.scales_real[env_names[0]][self.yaxis]]
        figures = create_scatter_fig(plot_data, scale_names)
        add_rating_background(figures, rating_pos, rating_mode)
        return figures

    def update_sliders(self, xaxis=None, yaxis=None, uploaded_scales=None):
        if uploaded_scales is not None:
            scales_dict = json.loads(base64.b64decode(uploaded_scales.split(',')[-1]))
            self.scales = load_scale(scales_dict)
        self.xaxis = xaxis or self.xaxis
        self.yaxis = yaxis or self.yaxis
        values = []
        for axis in [self.xaxis, self.yaxis]:
            all_ratings = [ sums[axis]['index'] for env_sums in self.summaries.values() for sums in env_sums[self.type] if sums[axis]['index'] is not None ]
            min_v = min(all_ratings) # if sl_idx == 0 else self.scales[axis][4 - sl_idx][1]
            max_v = max(all_ratings) # if sl_idx == 3 else self.scales[axis][3 - sl_idx][0]
            value = [entry[0] for entry in reversed(self.scales[axis][1:])]
            marks={ val: {'label': str(val)} for val in np.round(np.linspace(min_v, max_v, 10), 2)}
            # (sl_id, prop) for sl_id in ['scaleslider-x', 'scaleslider-y'] for prop in ['min', 'max', 'value', 'step', 'marks']]
            values.extend([min_v, max_v, value, marks])
        return values
    
    def update_scales(self, scale_slider_values):
        # check if sliders were updated from selecting axes, or if value was changed
        update_necessary = False
        for axis, values in zip([self.xaxis, self.yaxis], scale_slider_values):
            for sl_idx, sl_val in enumerate(values):
                if self.scales[axis][4 - sl_idx][0] != sl_val:
                    self.scales[axis][4 - sl_idx][0] = sl_val
                    self.scales[axis][3 - sl_idx][1] = sl_val
                    update_necessary = True
        if update_necessary:
            self.summaries, self.scales, self.scales_real = rate_results(self.summaries, self.reference_name, self.scales)

    def update_type(self, type=None):
        self.type = type or self.type
        options = [{'label': AXIS_NAMES[env], 'value': env} for env in self.keys[self.type]]
        xaxis = 'inference_power_draw' if self.type == 'inference' else 'train_power_draw'
        return options, xaxis, options, 'top1_val'

    def display_model(self, hover_data=None, env_names=None, rating_mode=None):
        if hover_data is not None:
            env_names = [list(self.summaries.keys())[0]] if env_names is None else env_names
            rating_mode = 'mean' if rating_mode is None else rating_mode
            point = hover_data['points'][0]
            if point['curveNumber'] % 2 == 0: # otherwise hovered on bars
                env_name = env_names[point['curveNumber'] // 2]
                self.current['summary'] = self.summaries[env_name][self.type][point['pointNumber']]
                self.current['logs'] = self.logs[env_name][self.type][point['pointNumber']]
                self.current['label'] = EnergyLabel(self.current['summary'], rating_mode)
            if self.current['summary'] is not None:
                return summary_to_str(self.current['summary'], rating_mode), self.current['label'].to_encoded_image()
        return 'no model summary to show', None

    def save_scales(self, save_labels_clicks=None):
        if save_labels_clicks is not None:
            return dict(content=save_scale(self.scales, None), filename=f'scales.json')

    def save_label(self, lbl_clicks=None, sum_clicks=None, log_clicks=None):
        if self.current['summary'] is not None:
            f_id = f'{self.current["summary"]["name"]}_{self.current["summary"]["environment"]}'
            if 'label' in dash.callback_context.triggered[0]['prop_id']:
                return dcc.send_bytes(self.current['label'].write(), filename=f'energy_label_{f_id}.pdf')
            elif 'sum' in dash.callback_context.triggered[0]['prop_id']:
                return dict(content=json.dumps(self.current['summary'], indent=4), filename=f'energy_summary_{f_id}.json')
            else: # full logs
                return dict(content=json.dumps(self.current['logs'], indent=4), filename=f'energy_logs_{f_id}.json')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Interactive energy index explorer")
    parser.add_argument("--directory", default='results', type=str, help="path directory with aggregated logs")
    args = parser.parse_args()

    app = Visualization(args.directory)
    app.run_server(debug=True)# , host='0.0.0.0', port=8888)
