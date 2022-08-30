import argparse
import base64
import json

import numpy as np
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc

from mlee.elex.pages import create_page
from mlee.elex.util import toggle_config, AXIS_NAMES, summary_to_html_table
from mlee.elex.graphs import create_scatter_fig, add_rating_background
from mlee.ratings import load_boundaries, save_boundaries, calculate_optimal_boundaries, save_weights, update_weights
from mlee.ratings import load_results, rate_results, calculate_compound_rating, TASK_METRICS_CALCULATION, MODEL_INFO
from mlee.label_generator import EnergyLabel


class Visualization(dash.Dash):

    def __init__(self, results_directory, reference_name='ResNet101', **kwargs):
        super().__init__(__name__, **kwargs)
        self.logs, summaries = load_results(results_directory)
        self.summaries, self.boundaries, self.boundaries_real = rate_results(summaries, reference_name)
        self.environments = {task: sorted(list(envs.keys())) for task, envs in self.summaries.items()}
        self.keys = {task: list(vals.keys()) for task, vals in TASK_METRICS_CALCULATION.items()}
        self.task, self.xaxis, self.yaxis = 'inference', 'top1_val', 'inference_power_draw'
        self.reference_name = reference_name
        self.current = { 'summary': None, 'label': None, 'logs': None }
        # setup page and create callbacks
        self.layout = create_page(list(self.keys.keys()))

        self.callback(
            Output("task-config", "is_open"),
            Input("btn-open-task-config", "n_clicks"),
            [State("task-config", "is_open")],
        ) (toggle_config)
        self.callback(
            [Output('x-weight', 'value'), Output('y-weight', 'value')],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('weights-upload', 'contents')]
        ) (self.update_metric_fields)
        self.callback(
            [Output('environments', 'options'), Output('environments', 'value'), Output('xaxis', 'options'), Output('xaxis', 'value'), Output('yaxis', 'options'),  Output('yaxis', 'value')],
            Input('task-switch', 'value')
        ) (self.update_task)
        self.callback(
            [Output(sl_id, prop) for sl_id in ['boundary-slider-x', 'boundary-slider-y'] for prop in ['min', 'max', 'value', 'marks']],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('boundaries-upload', 'contents'), Input('btn-calc-boundaries', 'n_clicks')]
        ) (self.update_boundary_sliders)
        self.callback(
            Output('figures', 'figure'),
            [Input('environments', 'value'), Input('scale-switch', 'value'), Input('rating', 'value'), Input('x-weight', 'value'), Input('y-weight', 'value'), Input('boundary-slider-x', 'value'), Input('boundary-slider-y', 'value')]
        ) (self.update_figures)
        self.callback(
            [Output('model-table', 'children'), Output('model-label', "src"), Output('btn-open-paper', "href")],
            Input('figures', 'hoverData'), State('environments', 'value'), State('rating', 'value')
        ) (self.display_model)
        self.callback(Output('save-label', 'data'), [Input('btn-save-label', 'n_clicks'), Input('btn-save-summary', 'n_clicks'), Input('btn-save-logs', 'n_clicks')]) (self.save_label)
        self.callback(Output('save-boundaries', 'data'), Input('btn-save-boundaries', 'n_clicks')) (self.save_boundaries)
        self.callback(Output('save-weights', 'data'), Input('btn-save-weights', 'n_clicks')) (self.save_weights)


    def update_figures(self, env_names=None, scale_switch=None, rating_mode=None, xweight=None, yweight=None, *slider_args):
        if xweight is not None and 'x-weight' in dash.callback_context.triggered[0]['prop_id']:
            self.summaries = update_weights(self.summaries, xweight, self.xaxis)
        if yweight is not None and 'y-weight' in dash.callback_context.triggered[0]['prop_id']:
            self.summaries = update_weights(self.summaries, yweight, self.yaxis)
        if any(slider_args) and 'slider' in dash.callback_context.triggered[0]['prop_id']:
            self.update_boundaries(slider_args)
        env_names = self.environments[self.task] if env_names is None else env_names
        scale_switch = 'index' if scale_switch is None else scale_switch
        rating_mode = 'mean' if rating_mode is None else rating_mode
        plot_data = {}
        for env in env_names:
            env_data = { 'names': [], 'ratings': [], 'x': [], 'y': [] }
            for sum in self.summaries[self.task][env]:
                env_data['names'].append(sum['name'])
                env_data['ratings'].append(calculate_compound_rating(sum, rating_mode))
                if scale_switch == 'index':
                    env_data['x'].append(sum[self.xaxis]['index'] or 0)
                    env_data['y'].append(sum[self.yaxis]['index'] or 0)
                else:
                    env_data['x'].append(sum[self.xaxis]['value'] or 0)
                    env_data['y'].append(sum[self.yaxis]['value'] or 0)
            plot_data[env] = env_data
        axis_names = [AXIS_NAMES[self.xaxis], AXIS_NAMES[self.yaxis]]
        if scale_switch == 'index':
            rating_pos = [self.boundaries[self.xaxis], self.boundaries[self.yaxis]]
            axis_names = [name.split('[')[0].strip() + ' Index' for name in axis_names]
        else:
            rating_pos = [self.boundaries_real[self.task][env_names[0]][self.xaxis], self.boundaries_real[self.task][env_names[0]][self.yaxis]]
        figures = create_scatter_fig(plot_data, axis_names)
        add_rating_background(figures, rating_pos, rating_mode)
        return figures

    def update_boundary_sliders(self, xaxis=None, yaxis=None, uploaded_boundaries=None, calculated_boundaries=None):
        if uploaded_boundaries is not None:
            boundaries_dict = json.loads(base64.b64decode(uploaded_boundaries.split(',')[-1]))
            self.boundaries = load_boundaries(boundaries_dict)
            self.summaries, self.boundaries, self.boundaries_real = rate_results(self.summaries, self.reference_name, self.boundaries)
        if calculated_boundaries is not None and 'calc' in dash.callback_context.triggered[0]['prop_id']:
            self.boundaries = calculate_optimal_boundaries(self.summaries, [0.8, 0.6, 0.4, 0.2])
            self.summaries, self.boundaries, self.boundaries_real = rate_results(self.summaries, self.reference_name, self.boundaries)
        self.xaxis = xaxis or self.xaxis
        self.yaxis = yaxis or self.yaxis
        values = []
        for axis in [self.xaxis, self.yaxis]:
            all_ratings = [ sums[axis]['index'] for env_sums in self.summaries[self.task].values() for sums in env_sums if sums[axis]['index'] is not None ]
            min_v = min(all_ratings) # if sl_idx == 0 else self.boundaries[axis][4 - sl_idx][1]
            max_v = max(all_ratings) # if sl_idx == 3 else self.boundaries[axis][3 - sl_idx][0]
            value = [entry[0] for entry in reversed(self.boundaries[axis][1:])]
            marks={ val: {'label': str(val)} for val in np.round(np.linspace(min_v, max_v, 10), 2)}
            # (sl_id, prop) for sl_id in ['boundary-slider-x', 'boundary-slider-y'] for prop in ['min', 'max', 'value', 'step', 'marks']]
            values.extend([min_v, max_v, value, marks])
        return values
    
    def update_boundaries(self, boundary_slider_values):
        # check if sliders were updated from selecting axes, or if value was changed
        update_necessary = False
        for axis, values in zip([self.xaxis, self.yaxis], boundary_slider_values):
            for sl_idx, sl_val in enumerate(values):
                if self.boundaries[axis][4 - sl_idx][0] != sl_val:
                    self.boundaries[axis][4 - sl_idx][0] = sl_val
                    self.boundaries[axis][3 - sl_idx][1] = sl_val
                    update_necessary = True
        if update_necessary:
            self.summaries, self.boundaries, self.boundaries_real = rate_results(self.summaries, self.reference_name, self.boundaries)

    def update_task(self, type=None):
        self.task = type or self.task
        avail_envs = [{"label": env, "value": env} for env in self.environments[self.task]]
        options = [{'label': AXIS_NAMES[env], 'value': env} for env in self.keys[self.task]]
        self.xaxis = 'inference_power_draw' if self.task == 'inference' else 'train_power_draw'
        self.yaxis = 'top1_val'
        return avail_envs, [avail_envs[0]['value']], options, self.xaxis, options, self.yaxis

    def display_model(self, hover_data=None, env_names=None, rating_mode=None):
        if hover_data is not None:
            # if not isinstance(env_names, list):
            #     env_names = [env_names]
            rating_mode = 'mean' if rating_mode is None else rating_mode
            point = hover_data['points'][0]
            if point['curveNumber'] % 2 == 0: # otherwise hovered on bars
                env_name = env_names[point['curveNumber'] // 2]
                self.current['summary'] = self.summaries[self.task][env_name][point['pointNumber']]
                self.current['logs'] = self.logs[self.task][env_name][point['pointNumber']]
                self.current['label'] = EnergyLabel(self.current['summary'], rating_mode)
            
            if self.current['summary'] is not None:
                html_table = summary_to_html_table(self.current['summary'], rating_mode)
                enc_label = self.current['label'].to_encoded_image()
                link = MODEL_INFO[self.current['summary']['name']]['url']
                return html_table,  enc_label, link
        return 'no model summary to show', None, "/"

    def save_boundaries(self, save_labels_clicks=None):
        if save_labels_clicks is not None:
            return dict(content=save_boundaries(self.boundaries, None), filename='boundaries.json')

    def update_metric_fields(self, xaxis=None, yaxis=None, upload=None):
        if upload is not None:
            weights = json.loads(base64.b64decode(upload.split(',')[-1]))
            self.summaries = update_weights(self.summaries, weights)
        any_summary = list(self.summaries[self.task].values())[0][0]
        return any_summary[self.xaxis]['weight'], any_summary[self.yaxis]['weight']

    def save_weights(self, save_weights_clicks=None):
        if save_weights_clicks is not None:
            return dict(content=save_weights(self.summaries, None), filename='weights.json')

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
    parser.add_argument("--host", default='localhost', type=str, help="default host")
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")
    args = parser.parse_args()

    app = Visualization(args.directory, external_stylesheets=[dbc.themes.COSMO])
    app.run_server(debug=args.debug, host=args.host, port=args.port)# , host='0.0.0.0', port=8888)
