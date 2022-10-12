from sqlite3 import Row
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
    label_display = html.Div(children=[
        html.H3('Energy Label:'),
        html.Img(id='model-label', className="img-fluid"),
        dbc.Tooltip("Click to enlarge", target="model-label"),
    ])

    label_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Energy Label")),
            dbc.ModalBody(html.Img(id='label-modal-img', style={"height": "70vh"})),
            dbc.ModalFooter(dbc.Button("Save Label", id="btn-save-label2"))
        ],
        id="label-modal",
        is_open=False,
    )

    table_model = html.Div(children=[
        html.H3('General Information:'),
        dbc.Table(id='model-table', bordered=True)
    ])
    table_metrics = html.Div(children=[
        html.H3('Efficiency Information:'),
        dbc.Table(id='metric-table', bordered=True),
    ])

    buttons = dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Button("Open Paper", id="btn-open-paper", class_name='col-12'), width=3),
            dbc.Col(dbc.Button("Save Label", id="btn-save-label", class_name='col-12'), width=3),
            dbc.Col(dbc.Button("Save Summary", id="btn-save-summary", class_name='col-12'), width=3),
            dbc.Col(dbc.Button("Save Logs", id="btn-save-logs", class_name='col-12'), width=3)
        ], style={'height': '50px'}, className="g-1"),
        # config buttons
        dbc.Row([
            dbc.Col(dbc.Button("Task Configuration", id="btn-open-task-config", size="lg", class_name='col-12'), width=6),
            dbc.Col(dbc.Button("Graph Configuration", id="btn-open-graph-config", size="lg", class_name='col-12'), width=6)
        ], style=dict(height='50px'), className="g-1")
    ], style={'padding': '0 0'})

    info_hover = dbc.Alert(
        "Hover over data points to show model information",
        id="info-hover",
        dismissable=True,
        is_open=True,
        color="info",
    )
    
    row0 = [
        dbc.Col(html.H1('ELEx - AI Energy Label Exploration')),
        dbc.Col(html.Img(src="assets/lamarr-logo.svg", className="img-fluid"), width=4)
    ]

    row1 = [
        dbc.Col(graph_scatter, width=8),
        dbc.Col(graph_bars, width=4)
    ]

    row2 = [
        dbc.Col([table_model, buttons], width=5),
        dbc.Col(table_metrics, width=5),
        dbc.Col(label_display, width=2)
    ]
    
    return html.Div([
        dbc.Container([
            dbc.Row(row0, style={"height": "15vh"}, align="center"),
            dbc.Row(row1, style={"height": "40vh"}),
            dbc.Row(info_hover),
            dbc.Row(row2, style={"height": "50vh"}),
        ]),
        # additional hidden html elements
        label_modal, task_configuration, graph_configuration, dcc.Download(id="save-label"),
    ])
