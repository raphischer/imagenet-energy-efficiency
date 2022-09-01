from dash import html, dcc
import dash_bootstrap_components as dbc


def create_axis_option(x=True):
    xy = 'x' if x else 'y'
    content = [
        html.Div(children=[dcc.Dropdown(id=f'{xy}axis')]),
        html.Label('Weight:'),
        dcc.Input(id=f"{xy}-weight", type='number', min=0, max=1, step=0.1),
        html.Label('Boundaries:'),
        dcc.RangeSlider(id=f'boundary-slider-{xy}', min=0, max=1,
            value=[.2, .4, .6, .8], step=.01, pushable=.01,
            tooltip={"placement": "bottom", "always_visible": True})  
    ]

    return dbc.AccordionItem(content, title=f'{xy}-Axis Configuration')

style_btn_cfg = {
    'width': '90%',
    'textAlign': 'center'
}

style_upload = dict({
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
}, **style_btn_cfg)


def create_page(tasks):
    
    # task configuration (offcanvas)
    task_configuration = dbc.Offcanvas(
        html.Div(children=[
            html.Div(children=[
                html.H2('ML Task:'),
                dbc.RadioItems(id='task-switch', value=tasks[0], options=[{'label': restype.capitalize(), 'value': restype} for restype in tasks],)
            ]),
            html.Div(children=[
                html.H2('Environments:'),
                dbc.Checklist(id='environments')
            ]),
        ]),
        id="task-config", title="Task Configuration", is_open=False,
    )
    btn_task_config = dbc.Button("Task Configuration", id="btn-open-task-config", n_clicks=0)

    # graph configuration (offcanvas)
    graph_configuration = dbc.Offcanvas(
        dbc.Accordion([
            create_axis_option(),
            create_axis_option(False),
            dbc.AccordionItem([
                html.H4('Scales'),
                dbc.RadioItems(
                    id='scale-switch', value='index',
                    options=[
                        {'label': 'Index Scale', 'value': 'index'},
                        {'label': 'Value Scale', 'value': 'real'}
                    ],
                ),
                html.H4('Boundaries and Weights'),
                dbc.Button("Calculate Optimal Boundaries", id="btn-calc-boundaries", active=False, style=style_btn_cfg),
                dbc.Button("Save Current Boundaries", id="btn-save-boundaries", style=style_btn_cfg),
                dcc.Download(id="save-boundaries"),
                dcc.Upload(
                    id="boundaries-upload", className='btn btn-default', style=style_btn_cfg,
                    children=['Drop or ', html.A('Select a Boundaries File (.json)')],
                ),
                dbc.Button("Save Current Metric Weights", id="btn-save-weights", style=style_btn_cfg),
                dcc.Download(id="save-weights"),
                dcc.Upload(
                    id="weights-upload", className='btn btn-default', style=style_btn_cfg,
                    children=['Drop or ', html.A('Select a Weights File (.json)')],
                ),
                html.H4('Rating Mode'),
                dbc.RadioItems(
                    id='rating', value='optimistic median',
                    options=[{'label': opt, 'value': opt.lower()} for opt in ['Optimistic Median', 'Pessimistic Median', 'Optimistic Mean', 'Pessimistic Mean', 'Best', 'Worst']],
                )
            ], title = 'More Graph Options')
        ], start_collapsed=True), id="graph-config", title="Graph Configuration", is_open=False, style=dict(width='40%')
    )
    btn_graph_config = dbc.Button("Graph Configuration", id="btn-open-graph-config", n_clicks=0)
    config_buttons = html.Div(children=[btn_task_config, btn_graph_config])

    # graphs
    graph_scatter = dcc.Graph(
        id='graph-scatter',
        responsive=True,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )
    graph_bars = dcc.Graph(
        id='graph-bars',
        responsive=True,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )
    
    # label display & tables
    # label_display = dbc.Card(
    #     [
    #         dbc.CardImg(id='model-label', top=True),
    #         dbc.CardBody([dbc.Button("Save Label", id="btn-save-label")]),
    #     ]
    # )    
    label_display = html.Div(children=[
        html.Img(id='model-label', style={"height": "40vh"})
    ])
    table_model = html.Div(children=[
        html.H2('General Information:'),
        dbc.Table(id='model-table', bordered=True)
    ])
    table_metrics = html.Div(children=[
        html.H2('Efficiency Information:'),
        dbc.Table(id='metric-table', bordered=True),
    ])
    buttons = html.Div(children=[
        dbc.Button("Open Paper", id="btn-open-paper"),
        dbc.Button("Save Label", id="btn-save-label"),
        dbc.Button("Save Summary", id="btn-save-summary"),
        dbc.Button("Save Logs", id="btn-save-logs")
    ])

    info_hover = dbc.Alert(
        "Hover over data points to show model information",
        id="info-hover",
        dismissable=True,
        is_open=True,
        color="info",
    )

    row1 = [
        dbc.Col(graph_scatter, width=8),
        dbc.Col(graph_bars, width=4)
    ]

    row2 = [
        dbc.Col([table_model, buttons, config_buttons], width=5),
        dbc.Col(table_metrics, width=5),
        dbc.Col(label_display, width=2)
    ]
    
    return html.Div([
        dbc.Row(html.H1('ELEx - AI Energy Labeling Exploration Tool')),
        dbc.Row(row1, style={"height": "50vh"}),
        dbc.Row(row2, style={"height": "50vh"}),
        dbc.Row(info_hover),
        # additional hidden html elements
        task_configuration, graph_configuration, dcc.Download(id="save-label"),
    ])
