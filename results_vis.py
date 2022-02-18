import base64
import json
import os

import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from plotly import colors
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from torch import save

from mlel.ratings import load_results, load_scale, save_scale, rate_results, aggregate_rating, KEYS


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLOR_SCALE = colors.make_colorscale(['rgb(0,255,0)', 'rgb(255,255,0)', 'rgb(255,0,0))'])
RATING_COLORS = colors.sample_colorscale(RATING_COLOR_SCALE, samplepoints=[float(c) / (4) for c in range(5)])
AXIS_NAMES = {
    "parameters": "Number of Parameters",
    "fsize": "Model File Size [B]", 
    "power_draw": "Power Draw / Sample [Ws]",
    "inference_time": "Inference Time / Sample [ms]",
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
        if isinstance(val, dict):
            ret_str.append(f'{AXIS_NAMES[key]:<30}: {val["value"]:<13.3f} - Index {val["index"]:4.2f} - Rating {val["rating"]}')
    full_str = '\n'.join(ret_str)
    # print(full_str)
    return full_str


def summary_to_label(summary, rating_mode):
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import black
    import fitz # PyMuPDF

    C_SIZE = (1560, 2411)
    canvas = Canvas("result.pdf", pagesize=C_SIZE)
    POS_TEXT = {
        "name":             (canvas.drawString,      90, '-Bold', .05, .83),
        "environment":      (canvas.drawString,      90, '',      .05, .41),
        "dataset":          (canvas.drawRightString, 90, '',      .95, .83),
        "power_draw":       (canvas.drawRightString, 90, '-Bold', .44, .35),
        "parameters":       (canvas.drawRightString, 68, '-Bold', .73, .17),
        "inference_time":   (canvas.drawRightString, 68, '-Bold', .19, .17),
        "top1_val":         (canvas.drawRightString, 68, '-Bold', .51, .06),
    }
    POS_RATINGS = { char: (.66, y) for char, y in zip('ABCDE', reversed(np.linspace(.461, .727, 5))) }
    frate = aggregate_rating(summary, rating_mode, 'ABCDE')
    canvas.drawInlineImage(os.path.join("label_design", "parts", "bg.png"), 0, 0)
    canvas.drawInlineImage(os.path.join("label_design", "parts", f"Rating_{frate}.png"), POS_RATINGS[frate][0] * C_SIZE[0], POS_RATINGS[frate][1] * C_SIZE[1])
    canvas.setFillColor(black)
    canvas.setLineWidth(3) # add stroke to make even bigger letters
    canvas.setStrokeColor(black)
    text=canvas.beginText()
    text.setTextRenderMode(2)
    canvas._code.append(text.getCode())
    # draw text parts
    for key, (draw_method, fsize, style, x, y) in POS_TEXT.items():
        text = summary[key] if isinstance(summary[key], str) else f'{summary[key]["value"]:3.2f}'
        canvas.setFont('Helvetica' + style, fsize)
        draw_method(int(C_SIZE[0] * x), int(C_SIZE[1] * y), text)
    pdf_doc = fitz.Document(stream=canvas.getpdfdata(), filetype='pdf')
    label_bytes = pdf_doc.load_page(0).get_pixmap().tobytes()
    base64_enc = base64.b64encode(label_bytes).decode('ascii')
    return 'data:image/png;base64,{}'.format(base64_enc), pdf_doc


def create_scatter_fig(scatter_pos, axis_title, names, env_names, ratings, ax_border=0.1):
    fig = make_subplots(rows=1, cols=2)
    posx, posy = scatter_pos
    for env_i, (x, y, name, env_name, rating) in enumerate(zip(posx, posy, names, env_names, ratings)):
        fig.add_trace(go.Scatter(
            x=x, y=y, text=name, mode='markers', name=env_name, marker_symbol=ENV_SYMBOLS[env_i],
            legendgroup=env_name, marker=dict(color=[RATING_COLORS[r] for r in rating], size=15),
            marker_line=dict(width=3, color='black')), row=1, col=1
        )
        counts = np.bincount(rating)
        fig.add_trace(go.Bar(
            name=env_name, x=['A', 'B', 'C', 'D', 'E'], y=counts, legendgroup=env_name,
            marker_pattern_shape=PATTERNS[env_i], marker_color=RATING_COLORS), row=1, col=2)
    fig.update_traces(textposition='top center', row=1, col=1)
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
    # fig.update_layout(legend=dict(x=.1, y=-.2, orientation="h"))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig.update_layout(barmode='stack')
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

    def __init__(self, results_directory, reference_name='ResNet101'):
        super().__init__(__name__)
        self.logs, summaries = load_results(results_directory)
        self.summaries, self.scales, self.scales_real = rate_results(summaries, reference_name)
        self.xaxis, self.yaxis = 'top1_val', 'power_draw'
        self.reference_name = reference_name
        self.current = { 'summary': None, 'label': None, 'logs': None }
        # setup page and create callbacks
        self.layout = self.create_page()
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
                        {'label': AXIS_NAMES[env], 'value': env} for env in KEYS
                    ]
                ),
                dcc.RangeSlider(id='scaleslider-x', min=0, max=1, value=[.2, .4, .6, .8], step=.01, pushable=.01, tooltip={"placement": "bottom", "always_visible": True})
            ]),
            html.Div(children=[
                html.H2('Y-Axis:'),
                dcc.Dropdown(
                    id='yaxis', value=self.yaxis,
                    options=[
                        {'label': AXIS_NAMES[env], 'value': env} for env in KEYS
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
                    id='rating', value='mean',
                    options=[{'label': opt, 'value': opt.lower()} for opt in ['Mean', 'Best', 'Worst', 'Majority', 'Median']],
                )
            ])
        ])

    def update_figures(self, env_names=None, scale_switch=None, rating_mode=None, *slider_args):
        if any(slider_args) and 'slider' in dash.callback_context.triggered[0]['prop_id']:
            self.update_scales(slider_args)
        env_names = [list(self.summaries.keys())[0]] if env_names is None else env_names
        scale_switch = 'index' if scale_switch is None else scale_switch
        rating_mode = 'mean' if rating_mode is None else rating_mode
        x, x_ind, y, y_ind, ratings, names = [], [], [], [], [], [],
        for env in env_names:
            names.append([r['name'] for r in self.summaries[env]])
            x.append([r[self.xaxis]['value'] for r in self.summaries[env]])
            y.append([r[self.yaxis]['value'] for r in self.summaries[env]])
            x_ind.append([r[self.xaxis]['index'] for r in self.summaries[env]])
            y_ind.append([r[self.yaxis]['index'] for r in self.summaries[env]])            
            ratings.append([aggregate_rating(summary, rating_mode) for summary in self.summaries[env]])
        scale_names = [AXIS_NAMES[self.xaxis], AXIS_NAMES[self.yaxis]]
        if scale_switch == 'index':
            scatter_pos = [x_ind, y_ind]
            rating_pos = [self.scales[self.xaxis], self.scales[self.yaxis]]
            scale_names = [name.split('[')[0].strip() + ' Index' for name in scale_names]
        else:
            scatter_pos = [x, y]
            rating_pos = [self.scales_real[env_names[0]][self.xaxis], self.scales_real[env_names[0]][self.yaxis]]
        figures = create_scatter_fig(scatter_pos, scale_names, names, env_names, ratings)
        add_rating_background(figures, rating_pos, rating_mode)
        return figures

    def update_sliders(self, xaxis=None, yaxis=None, uploaded_scales=None):
        if uploaded_scales is not None:
            scales_dict = json.loads(base64.b64decode(uploaded_scales.split(',')[-1]))
            self.scales = load_scale(scales_dict)
        # [Output(sl_id, prop) for sl_id in ['scaleslider-x', 'scaleslider-y'] for prop in ['min', 'max', 'value', 'step', 'marks']],
        self.xaxis = xaxis or self.xaxis
        self.yaxis = yaxis or self.yaxis
        values = []
        for axis in [self.xaxis, self.yaxis]:
            all_ratings = [ sums[axis]['index'] for env_sums in self.summaries.values() for sums in env_sums ]
            min_v = min(all_ratings) # if sl_idx == 0 else self.scales[axis][4 - sl_idx][1]
            max_v = max(all_ratings) # if sl_idx == 3 else self.scales[axis][3 - sl_idx][0]
            value = [entry[0] for entry in reversed(self.scales[axis][1:])]
            marks={ val: {'label': str(val)} for val in np.round(np.linspace(min_v, max_v, 10), 2)}
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

    def display_model(self, hover_data=None, env_names=None, rating_mode=None):
        if hover_data is not None:
            env_names = [list(self.summaries.keys())[0]] if env_names is None else env_names
            rating_mode = 'mean' if rating_mode is None else rating_mode
            point = hover_data['points'][0]
            if point['curveNumber'] % 2 == 0: # otherwise hovered on bars
                env_name = env_names[point['curveNumber'] // 2]
                self.current['summary'] = self.summaries[env_name][point['pointNumber']]
                self.current['logs'] = self.logs[env_name][point['pointNumber']]
                self.current['label_img'], self.current['label'] = summary_to_label(self.current['summary'], rating_mode)
            if self.current['summary'] is not None:
                return summary_to_str(self.current['summary'], rating_mode), self.current['label_img']
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

    app = Visualization('results')
    app.run_server(debug=True)# , host='0.0.0.0', port=8888)