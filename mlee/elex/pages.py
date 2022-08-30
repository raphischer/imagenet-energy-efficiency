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


def create_page(tasks):
    
    task_configuration = dbc.Offcanvas(
        html.Div(children=[
            html.Div(children=[
                html.H2('ML Task:'),
                dbc.RadioItems(id='task-switch', value=tasks[0],
                    options=[{'label': restype.capitalize(), 'value': restype} for restype in tasks],)
            ]),
            html.Div(children=[
                html.H2('Environments:'),
                dbc.Checklist(id='environments')
            ]),
        ]),
        id="task-config",
        title="Task Configuration",
        is_open=False,
    )
    btn_task_config = dbc.Button("Task Configuration", id="btn-open-task-config", n_clicks=0)

    graph = dcc.Graph(
        id='figures',
        responsive=True,
        config={'responsive': True},
        style={'height': '100%', 'width': '100%'}
    )

    axis_configuration = dbc.Accordion([
        create_axis_option(),
        create_axis_option(False),
        dbc.AccordionItem([
            dbc.RadioItems(
                id='scale-switch', value='index',
                options=[
                    {'label': 'Reference Index', 'value': 'index'},
                    {'label': 'Real Values', 'value': 'real'}
                ],
            ),
            dbc.Button("Calculate Optimal Boundaries", id="btn-calc-boundaries", active=False),
            dbc.Button("Save Current Boundaries", id="btn-save-boundaries"),
            dcc.Download(id="save-boundaries"),
            dcc.Upload(
                id="boundaries-upload",
                children=['Drag and Drop or ', html.A('Select a Boundaries File (.json)')],
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
            dbc.Button("Save Current Metric Weights", id="btn-save-weights"),
            dcc.Download(id="save-weights"),
            dcc.Upload(
                id="weights-upload",
                children=['Drag and Drop or ', html.A('Select a Weights File (.json)')],
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
            html.Div(children=[
                html.H2('Rating mode:'),
                dbc.RadioItems(
                    id='rating', value='optimistic median',
                    options=[{'label': opt, 'value': opt.lower()} for opt in ['Optimistic Median', 'Pessimistic Median', 'Optimistic Mean', 'Pessimistic Mean', 'Best', 'Worst']],
                )
            ])
        ], title = 'More axis options')
    ], start_collapsed=True)
    
    label_display = dbc.Card(
        [
            dbc.CardImg(id='model-label', top=True),
            dbc.CardBody(
                [
                    dbc.Button("Save Label", id="btn-save-label")
                ]
            ),
        ]
        # style={"width": "18rem"},
    )

    
    
    # html.Div(children=[
    #     html.Img(id='model-label', style={'height': '400px'})
    # ])

    table_display = html.Div(children=[
        dbc.Table(id='model-table', bordered=True),
    ])

    btn_open_paper = dbc.Button("Open Paper", id="btn-open-paper")
    btn_save_summary = dbc.Button("Save Summary", id="btn-save-summary")
    btn_save_logs = dbc.Button("Save Logs", id="btn-save-logs")

    row1 = [
        dbc.Col(btn_task_config, width=5),
    ]

    row2 = [
        dbc.Col(graph, width=5),
        dbc.Col(label_display, width=2),
        dbc.Col([table_display, btn_open_paper, btn_save_summary, btn_save_logs], width=5)
    ]
    
    return html.Div([
        dbc.Row(row1),
        dbc.Row(row2, style={"height": "75vh"}),
        dbc.Row(axis_configuration),
        dbc.Row(task_configuration),
        dbc.Row(dcc.Download(id="save-label"))
    ])
