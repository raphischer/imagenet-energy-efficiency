from plotly import colors
from plotly.validators.scatter.marker import SymbolValidator
from dash import html
import dash

from mlee.ratings import calculate_compound_rating


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLOR_SCALE = colors.make_colorscale(['rgb(0,255,0)', 'rgb(255,255,0)', 'rgb(255,0,0))'])
RATING_COLORS = colors.sample_colorscale(RATING_COLOR_SCALE, samplepoints=[float(c) / (4) for c in range(5)])
AXIS_NAMES = {
    "parameters":               "M Parameters [#]",
    "gflops":                   "(Giga) Floating Point Operations [#]",
    "fsize":                    "Model File Size [B]", 
    "inference_power_draw":     "Inference Power Draw / Batch [Ws]",
    "inference_time":           "Inference Time / Batch [ms]",
    "train_power_draw":         "Full Training Power Draw [Ws]",
    "train_power_draw_epoch":   "Training Power Draw per Epoch [kWh]",
    "train_time":               "Full Training Time [h]",
    "train_time_epoch":         "Training Time per Epoch [h]",
    "top1_val":                 "Top-1 Validation Accuracy [%]",
    "top5_val":                 "Top-5 Validation Accuracy [%]"
}
PATTERNS = ["", "/", ".", "x", "-", "\\", "|", "+", "."]


def summary_to_str(summary, rating_mode):
    final_rating = calculate_compound_rating(summary, rating_mode)
    environment = f"({summary['environment']} Environment)"
    ret_str = [f'Name: {summary["name"]:17} {environment:<34} - Final Rating {final_rating}']
    for key, val in summary.items():
        if isinstance(val, dict) and "value" in val:
            if val["value"] is None:
                value, index = f'{"n.a.":<13}', "n.a."
            else:
                value, index = f'{val["value"]:<13.3f}', f'{val["index"]:4.2f}'
            ret_str.append(f'{AXIS_NAMES[key]:<30}: {value} - Index {index} - Rating {val["rating"]}')
    full_str = '\n'.join(ret_str)
    return full_str


def summary_to_html_tables(summary, rating_mode):
    final_rating = calculate_compound_rating(summary, rating_mode)
    info_header = [
        html.Thead(html.Tr([html.Th("Task"), html.Th("Model Name"), html.Th("Environment"), html.Th("Final Rating")]))
    ]
    
    task = f"{summary['task_type']} on {summary['dataset']}"
    info_row = [html.Tbody([html.Tr([html.Td(field) for field in [task, summary['name'], summary['environment'], final_rating]])])]

    metrics_header = [
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Index"), html.Th("Rating")]))
    ]
    metrics_rows = []
    for key, val in summary.items():
        if isinstance(val, dict) and "value" in val:
            if val["value"] is None:
                value, index = "n.a.", "n.a."
            else:
                value, index = val["value"], val["index"]
            metrics_rows.append(html.Tr([html.Td(field) for field in [AXIS_NAMES[key], value, index, val["rating"]]]))

    model = info_header + info_row
    metrics = metrics_header + [html.Tbody(metrics_rows)]
    return model, metrics



def toggle_element_visibility(n1, is_open):
    if n1:
        return not is_open
    return is_open

