import pandas as pd
from dash import Dash, html
from dash import dcc, no_update
import dash_cytoscape as cyto  # pip install dash-cytoscape
import dash_bootstrap_components as dbc
from datetime import date
from dash import dash_table
import visdcc  # pip install visdcc
from callbacks.NER_TimeEgo_callback import *



# In[]
# X2, raw_S2, CR_doc2, CO_doc2, CR_sen2, CO_sen2, XX_Doc2, XX_Sent2 = time_range_data(start_date2, end_date2)
# In[]
# 測試用
# Z = "Carnegie_Mellon_University"
# Z = "3D printing"  input_filter = ["com", "rocket", "loc"]
# Unit = "Sentence"  type = 'correlation' total_nodes_num = 10  threshold = 0.5 input_filter = ["com"]
# input_filter = "不篩選"
# In[]
# 計算edge寬度用





# 建立顏色和CLASS的對應dict
CLASS_COLOR_MAP = dict(zip(CLASS_LIST, COLOUR))

# 外部css元件
external_stylesheets = ['https://unpkg.com/antd@3.1.1/dist/antd.css',
                        'https://rawgit.com/jimmybow/CSS/master/visdcc/DataTable/Filter.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',
                        'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        ]

app = Dash(__name__, external_stylesheets=external_stylesheets)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',  # 邊框屬性
        'overflowX': 'scroll'  # 內容水平延伸
    }
}
merged_df = pd.DataFrame()
# set my legend
propotion = 100/len(COLOUR)
legend = []
for c, label in zip(COLOUR, LEGEND):
    l = html.Div(label,
                 style={
                     'background-color': c,
                     'padding': '20px',
                     'color': 'white',
                     'display': 'inline-block',
                     'width': str(propotion)+'%',
                     'font-size': '20px'
                 })
    legend.append(l)

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
all_docs = RAW_DOC.sort_values(by='date', ascending=False).reset_index(drop=True)
dataset_info = pd.DataFrame({
    "APP更新日期": [all_docs.iloc[0]['date']],
    "資料來源": [DATA_SOURCE_NAME],
    "資料筆數": [len(all_docs)],
    "資料時間": [f"{all_docs.iloc[-1]['date']}至{all_docs.iloc[0]['date']}"]
})

app.layout = html.Div(children=[
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;800&display=swap'
    ),
    html.H1(f"{NER_TOPIC} NER時間單中心網路分析",
            style={
                'font-size': '36px',
                'textAlign': 'center',
                'backgroundColor': '#daf5ed',
                'margin': '0px',
                'font-weight': 'bold',
                'padding': '5px'
            }
            ),
    html.H6('以特定主題為中心，選定文集之起始區段，從文集中選出相關性最高的關鍵詞，並對它們進行社會網路分析',
            style={
                'font-size': '24px',
                'textAlign': 'center',
                'backgroundColor': '#f2efe4',
                'padding': '3px',
                'margin': '0px',
    }
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

            # Time_ranger
            dbc.Label("輸入時間區間",
                      style=bold_orange),
            dcc.DatePickerRange(
                id='my-date-picker-range',
                style={'whiteSpace': 'pre-wrap'},
                min_date_allowed=date(
                    start_date_object.year, start_date_object.month, start_date_object.day),
                max_date_allowed=date(
                    latest_date_object.year, latest_date_object.month, latest_date_object.day),
                initial_visible_month=date(2024, 1, 1),
                start_date=date(2024, 1, 1),
                end_date=date(2024, 6, 30)
            ),
            html.Br(),
            html.Button(id='submit_button', n_clicks=1, children='Submit',
                            style={
                                'height': '38px',
                                'whiteSpace': 'pre-wrap',
                                'font-size': '16px',
                                'font-weight': 'bold',
                                'line-height': '38px',
                                'letter-spacing': '.1rem',
                                'color': '#555',
                                'backgroundColor': '#daf5ed',
                                'text-align': 'center',
                                #'background-color':'transparent',
                                'border-radius': '4px',
                                'border': '1px solid #bbb',
                                'padding': '0 30px',
                                'cursor': 'pointer',
                                'margin': '4px',
                                #'font-family': "Open Sans", 
                                'box-sizing': 'border-box',
                                }
                        # style={'font-size': '16px'}
                        ),
            html.Br(),
            dbc.Label("選擇關鍵字類別",
                      style=bold_orange),
            # 切換類別下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_class',
                value=3,
                clearable=False,
                options=[
                    {'label': CLASS_LIST[i], 'value': i}
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
                    {'label': CLASS_LIST[i], 'value': clas}
                    for i, clas in enumerate(CLASS_LIST)
                ],
                style={'margin': '0.5rem 0rem 0rem 0rem'}
            ),
            html.H6('針對網路圖的節點類別進行篩選',
                    style=annotation),

            dbc.Label("設定網路節點數量",
                      style=inline_orange),
            dcc.Dropdown(
                id='total_nodes_num',
                options=[{'label': str(i), 'value': i}
                         for i in range(3, 21)],
                value=4,
                style={
                    'verticalAlign': 'top',
                    'margin': '0rem 1.5rem 0rem 0rem',
                    'display': 'inline-block'
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
            # 計算方式選鈕
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
            html.Div(legend,
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

register_callback(app)

#app.run_server(debug=True, use_reloader=False)
# # Turn off reloader if inside Jupyter
app.run_server("0.0.0.0",port=8090, debug=True)