import pandas as pd
from dash import Dash, html
from dash import dcc, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
# import dash_bootstrap_components as dbc
import visdcc  # pip install visdcc
from dash import dash_table
from dash_style import *

from callbacks.NER_ego_callback import *
# In[]
# 字典

# 關鍵字類別
# 類別顏色
Sen_Doc_list = ["Sentence", "Document"]
# In[]


# 外部css元件
external_stylesheets = ['https://unpkg.com/antd@3.1.1/dist/antd.css',
                        'https://rawgit.com/jimmybow/CSS/master/visdcc/DataTable/Filter.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',
                        'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;800&display=swap'
                        ]

app = Dash(__name__, external_stylesheets=external_stylesheets)


merged_df = pd.DataFrame()

# 建立顏色和CLASS的對應dict
CLASS_COLOR_MAP = dict(zip(CLASS_LIST, COLOUR))

bold_orange = {
    'font-size': '16px',
    'color': '#CA774B',
    'font-weight': 'bold',
    'display': 'block',
    'margin': '1rem 0rem 0rem 0rem'}  # top,right,bottom,left

inline_orange = {
    'font-size': '16px',
    'color': '#CA774B',
    'font-weight': 'bold',
    'display': 'inline-block',
    'margin': '0.5rem 1.5rem 0rem 0rem'}

annotation = {'font-size': '14px', 'color': '#66828E'}
# 資料集資訊
RAW_DOC['date'] = pd.to_datetime(RAW_DOC['date']).dt.date
dataset_info = {
    "APP更新日期": RAW_DOC['date'].max().strftime('%Y-%m-%d'),
    "資料來源": DATA_SOURCE_NAME,
    "資料筆數": len(RAW_DOC),
    "資料時間": f"{RAW_DOC['date'].min().strftime('%Y-%m-%d')}至{RAW_DOC['date'].max().strftime('%Y-%m-%d')}"
}

app.layout = html.Div(children=[
    html.H1(f"{NER_TOPIC} NER單中心網路分析",
            style=STYLE_FOR_TITLE
            ),
    html.H6('以特定主題為中心，從文集中選出相關性最高的關鍵詞，並對它們進行社會網絡分析',
            style=STYLE_FOR_SUBTITLE
            ),
    html.Div([
        html.Div([
            # 資料集資訊
            html.Table(children=[
                html.Tr([
                    html.Td(key),
                    html.Td(value)
                ]) for key, value in dataset_info.items()
            ], id='dataset-info-table', style={'font-size': '10px', 'color': '#66828E'}),
            dbc.Label("選擇關鍵字類別",
                      style=bold_orange),
            # 切換類別下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_class',
                value=4,
                clearable=False,
                options=[
                    {'label': clas, 'value': i}
                    for i, clas in enumerate(CLASS_LIST)
                ],
                style={'margin': '0.5rem 0rem 0.8rem 0rem'}
            ),
            dbc.Label("選擇關鍵字",
                      style=bold_orange),
            # 選擇中心詞下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_name',
                value=DEAFAULT_VALUE,
                clearable=False,
                options=[
                    {'label': name, 'value': name}
                    for name in origin_key_dict_pd[origin_key_dict_pd['label'] == CLASS_LIST[0]]['keywords'].to_list()
                ],
                style={'margin': '0.5rem 0rem 0.8rem 0rem'}
            ),
            dbc.Label("網路篩選遮罩",
                      style=bold_orange),
            # 網路篩選遮罩下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_filter',
                clearable=False,
                multi=True,
                options=[
                    {'label': clas, 'value': clas}
                    for i, clas in enumerate(CLASS_LIST)
                ],
                style={'margin': '0.5rem 0rem 0rem 0rem'}
            ),
            html.H6('針對網路圖的節點類別進行篩選',
                    style=annotation),
            dbc.Label("關鍵字節點數量",
                      style=inline_orange),
            html.Br(),
            dbc.Label("一階",
                      style=inline_orange),
            dcc.Dropdown(
                id='stratum1_num',
                options=[{'label': str(i), 'value': i+1}
                         for i in range(3, 21)],
                value=8,
                style={
                    'verticalAlign': 'top',
                    'margin': '0rem 1.5rem 0rem 0rem',
                }
            ),
            dbc.Label("二階",
                      style=inline_orange),
            dcc.Dropdown(
                id='stratum2_num',
                options = [{'label': str(i), 'value': i+1}
                         for i in range(0, 11)],

                value=0,
                style={
                    'verticalAlign': 'top',
                    'margin': '0rem 1.5rem 0rem 0rem',
                }
            ),
            dbc.Label("依關聯強度篩選鏈結",
                      style=bold_orange),
            # 網路圖篩選節點閥值slider
            dcc.Slider(
                id="threshold_slide", min=0, max=1, step=0.01,
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
                marks={i/10: str(i/10) for i in range(51)},
                value=0.5
            ),
            dbc.Label("字詞連結段落", style=inline_orange),
            # 計算單位選鈕
            dcc.RadioItems(
                id='RadioItems_SenorDoc',
                options=[{'label': '句 ', 'value': 'Sentence'},
                         {'label': '篇', 'value': 'Document'},],
                value='Sentence',
                inline=True,
                style={'margin': '0.5rem 1rem 0rem 0rem',
                       'display': 'inline-block'}
            ),
            dbc.Label("連結強度計算方式",
                      style=bold_orange),
            dcc.RadioItems(
                id='RadioItems_CRorCO',
                options=[{'label': '共同出現次數', 'value': 'co-occurrence'},
                         {'label': '相關係數', 'value': 'correlation'},],
                value='correlation',
                inline=True,
                style={'margin': '0.5rem 0rem 0rem 0rem'}
            ),
            dbc.Label("連結強度依據字詞出現頻率", style=annotation),
            html.Br(),
            dbc.Label("較高，可選「相關係數」", style=annotation),
            html.Br(),
            dbc.Label("較低，可擇「共同出現次數」", style=annotation),
            dbc.Label("調整節點字體大小", style=bold_orange), 
            dcc.Slider(
                id='nodes-font-size-slider',
                min=10,
                max=40,
                step=2,
                value=20,
                marks={i: str(i) for i in range(10, 41, 10)}
            ), 
            dbc.Label("調整文章字體大小", style=bold_orange), 
            dcc.Slider(
                id='textarea-font-size-slider',
                min=10,
                max=40,
                step=2,
                value=20,
                marks={i: str(i) for i in range(10, 41, 10)}
            ), 
        ],
            style={
            'background-color': '#daf5ed',
            'display': 'inline-block',
            'width': '15%',
            'height': '1000px',
            'padding': '0.5%'}
        ),
        html.Div([
            # legend
            html.Div(get_ledgend(COLOUR, LEGEND),
                     style={
                         'background-color': "#ede7d1",
                         'color': '#f2efe4',
                         'height': '7.5%',
                         'text-align': 'center',
                         'font-size': '24px',
                         'padding': '0px'}),
            # 網路圖
            visdcc.Network(
                id='net',
                selection={'nodes': [], 'edges': []},
                options={
                    'interaction': {
                        'hover': True,
                        'tooltipDelay': 300,
                    },
                    'groups': {
                        class_name: {'color': color}
                        for class_name, color in CLASS_COLOR_MAP.items()
                    },
                    'autoResize': True,
                    'height': '800px',
                    'width': '100%',
                    'layout': {
                        'improvedLayout': True,
                        'hierarchical': {
                            'enabled': False,
                            'levelSeparation': 150,
                            'nodeSpacing': 100,
                            'treeSpacing': 200,
                            'blockShifting': True,
                            'edgeMinimization': True,
                            'parentCentralization': True,
                            'direction': 'UD',        # UD, DU, LR, RL
                            'sortMethod': 'hubsize'   # hubsize, directed
                        }
                    },
                    'physics': {
                        'enabled': True,
                        'barnesHut': {
                            'theta': 0.5,
                            'gravitationalConstant': -20000,  # repulsion強度
                            'centralGravity': 0.3,
                            'springLength': 95,
                            'springConstant': 0.04,
                            'damping': 0.09,
                            # 'avoidOverlap': 0.01
                        },
                    },
                    'adaptiveTimestep': True,
                }
            ),
        ], style={'display': 'inline-block',
                  'width': '50%',
                  'verticalAlign': 'top'}
        ),
        # 放置文章
        html.Div([
            # 文本元件
            dash_table.DataTable(
                id='doc_number',
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '20px',
                    'width': '30px',
                    'maxWidth': '200px',
                    'text-align': 'left',
                    'whiteSpace': 'normal',
                },
                style_table={'height': 'auto',
                             'overflowY': 'auto', 'fontSize': 20},
                columns=[
                    {"id": "doc_id_num", "name": "篇數"},
                    {"id": "recent_num", "name": "句數"},
                ],
            ),
            dash_table.DataTable(
                id='table',
                # css=[dict(selector="p", rule="margin: 0px; text-align: center")],
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '20px',
                    'width': '30px',
                    'maxWidth': '200px',
                    'text-align': 'left',
                    'whiteSpace': 'normal',
                },
                columns=[
                    {"id": "Date", "name": "Date"},
                    {"id": "id", "name": "Doc_id", "presentation": "markdown"},
                    {"id": "Recent", "name": "Recent", "presentation": "markdown"},
                ],
                markdown_options={"html": True},
                page_size=10,
                fixed_rows={'headers': True},),
            dash_table.DataTable(
                id='textarea_table',
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '20px',
                    'width': '30px',
                    'maxWidth': '200px',
                    'text-align': 'left',
                    'whiteSpace': 'normal',
                },
                style_table={'height': '500px',
                             'overflowY': 'auto', 'fontSize': 20},
                columns=[
                    {"id": "article", "name": "article",
                        "presentation": "markdown"},
                ],
                markdown_options={"html": True},
                editable=False,
            ),
        ], style={
            'background-color': COLOUR[0],
            'display': 'inline-block',
            'width': '35%',
            'height': '150%',
            'verticalAlign': 'top'}),
    ], style={'height': '100%', 'width': '100%'}),

], style={'font-family': 'Noto Sans TC, sans-serif'})



# In[]
# 測試用
# data = ['NASA']
# In[]
# node點擊事件


#def node_recation(Unit, data, type, total_nodes_num, threshold):

register_callback(app)

#app.run_server(debug=True, use_reloader=False)
app.run_server("0.0.0.0",port=8071, debug=True)