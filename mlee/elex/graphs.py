import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mlee.ratings import calculate_compound_rating
from mlee.elex.util import RATING_COLORS, ENV_SYMBOLS, PATTERNS

def add_rating_background(fig, rating_pos, mode):
    for xi, (x0, x1) in enumerate(rating_pos[0]):
        for yi, (y0, y1) in enumerate(rating_pos[1]):
            color = calculate_compound_rating([xi, yi], mode, RATING_COLORS)
            fig.add_shape(type="rect", layer='below', line=dict(color='#0c122b'), fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)


def create_scatter_graph(plot_data, axis_title, ax_border=0.1):
    fig = go.Figure()
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], text=data['names'], name=env_name, 
            mode='markers', marker_symbol=ENV_SYMBOLS[env_i],
            legendgroup=env_name, marker=dict(color=[RATING_COLORS[r] for r in data['ratings']], size=15),
            marker_line=dict(width=3, color='black'))
        )
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
    fig.update_layout(legend=dict(x=.5, y=1, orientation="h", xanchor="center", yanchor="bottom",))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig.update_layout(template='plotly_dark', paper_bgcolor="#0c122b", plot_bgcolor="#0c122b")
    min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
    min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
    diff_x, diff_y = max_x - min_x, max_y - min_y
    fig.update_layout(
        xaxis_range=[min_x - ax_border * diff_x, max_x + ax_border * diff_x],
        yaxis_range=[min_y - ax_border * diff_y, max_y + ax_border * diff_y],
        margin={'l': 10, 'r': 10, 'b': 10, 't': 10}
    )
    return fig

def create_bar_graph(plot_data):
    fig = go.Figure()
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        counts = np.bincount(data['ratings'])
        fig.add_trace(go.Bar(
            name=env_name, x=['A', 'B', 'C', 'D', 'E'], y=counts, legendgroup=env_name,
            marker_pattern_shape=PATTERNS[env_i], marker_color=RATING_COLORS, showlegend=False)
        )
    fig.update_layout(barmode='stack')
    fig.update_layout(margin={'l': 10, 'r': 10, 'b': 10, 't': 10})
    fig.update_layout(xaxis_title='Final Rating', yaxis_title='Number of Models')
    fig.update_layout(template='plotly_dark', paper_bgcolor="#0c122b", plot_bgcolor="#0c122b")
    return fig