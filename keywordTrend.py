from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

dtm = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\freq_DTM.csv')
monthList = ['一月', '二月', '三月', '四月', '五月',
             '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']
monthOption = []  # the list for option
for i in range(1, 13):
    if i//10 < 1:
        monthDict = {'label': monthList[i-1], 'value': '0' + str(i)}
    else:
        monthDict = {'label': monthList[i-1], 'value': str(i)}
    monthOption.append(monthDict)
# the last month of the data # str: '2023-08'
theLastMonth = list(dtm.columns.values)[-1]
# the first month of the data # str: '2017-01'
theFirstMonth = list(dtm.columns.values)[2]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='國防 SpaceNews 關鍵字詞聲量趨勢查詢', style={'textAlign': 'center'}),
    html.H2(children='關鍵字趨勢', style={
        'font-size': '28px', 'textAlign': 'center', "color": '#36451a'}),
    html.Div([
        html.Hr(style={'borderWidth': "1vh", "width": "100%",
                "borderColor": "#83A742", "opacity": "unset"}),
        html.H3(children='選擇類別', style={'textAlign': 'left'}),
        dcc.RadioItems(id='inputLabel',
                       options=[
                           {'label': '一般術語', 'value': 'term'},
                           {'label': '公司', 'value': 'com'},
                           {'label': '組織', 'value': 'org'},
                           {'label': '衛星', 'value': 'satellite'},
                           {'label': '國家地區', 'value': 'loc'},
                           {'label': '火箭', 'value': 'rocket'}
                       ],
                       value="loc",
                       inline=True
                       ),
        html.H3(children='選擇關鍵字排名區間', style={'textAlign': 'left'}),
        dcc.RangeSlider(min=1, max=50, step=1, value=[1, 10], tooltip={
                        "placement": "bottom", "always_visible": True}, id='K'),
        html.Div([
            html.Div([
                html.H3(children='起始日', style={
                        'width': '30%', 'display': 'inline-block'}),

                dcc.Dropdown(
                    options=[
                        {'label': '2017', 'value': '2017'},
                        {'label': '2018', 'value': '2018'},
                        {'label': '2019', 'value': '2019'},
                        {'label': '2020', 'value': '2020'},
                        {'label': '2021', 'value': '2021'},
                        {'label': '2022', 'value': '2022'},
                        {'label': '2023', 'value': '2023'},
                    ],
                    value='2017', id='year_from', style={'width': '30%', 'display': 'inline-block'}
                ),
                dcc.Dropdown(
                    options=monthOption,
                    value='01', id='month_from', style={'width': '30%', 'display': 'inline-block'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.H3(children='終止日', style={
                    'textAlign': 'left', 'width': '30%', 'display': 'inline-block'}),
                dcc.Dropdown(
                    options=[
                        {'label': '2017', 'value': '2017'},
                        {'label': '2018', 'value': '2018'},
                        {'label': '2019', 'value': '2019'},
                        {'label': '2020', 'value': '2020'},
                        {'label': '2021', 'value': '2021'},
                        {'label': '2022', 'value': '2022'},
                        {'label': '2023', 'value': '2023'},
                    ],
                    value='2017', id='year_to', style={'width': '30%', 'display': 'inline-block'}
                ),
                dcc.Dropdown(
                    options=monthOption,
                    value='01', id='month_to', style={'width': '30%', 'display': 'inline-block'}
                )
            ], style={'width': '50%', 'display': 'inline-block'})

        ], style={'margin': '20px'}),
        html.Hr(style={'borderWidth': "1px", "width": "100%",
                "borderColor": "#e7f5d0"}),
    ], style={'color': '#83A742'}),
    html.Div([
        html.Div([
            html.U(children='每月關鍵字出現次數', style={
                'font-weight': 'bold', 'font-size': '28px', 'textAlign': 'left', "color": '#36451a'}),
            dcc.Graph(id='line-chart'),
        ], style={'border-right': '1px solid #83A742'}),
        # html.Div([
        #     html.U(children='每月文章比重', style={
        #         'font-weight': 'bold', 'font-size': '28px', 'textAlign': 'left', "color": '#36451a'}),
        #     dcc.Graph(id='percentage'),], style={'width': '48%', 'display': 'inline-block', 'border-left': '9px solid white'})
    ], style={'margin': '20px'})
])


@app.callback(
    Output('month_from', 'options'),
    Output('month_from', 'value'),
    Input('year_from', 'value')
)
def update_month_from_options(selected_year):
    # theLastMonth[0:4]<- str:'2023-08'[0:4] > 2023
    lastYearOption_from = []
    if selected_year == theLastMonth[0:4]:
        # theLastMonth[5:7]<- str:'2023-08'[5:7] > int(08)-2 > 6 loop shows 6-1=5
        for i in range(1, int(theLastMonth[5:7])-2):
            if i//10 < 1:
                lastMonthDict = {
                    'label': monthList[i-1], 'value': '0' + str(i)}  # transform integer 1 to string 01
            else:
                lastMonthDict = {'label': monthList[i-1], 'value': str(i)}
            lastYearOption_from.append(lastMonthDict)
        options = lastYearOption_from
        value = '01'  # Set value to None by default if no option is selected
    else:
        options = monthOption
        value = '01'
    return options, value


@app.callback(
    Output('month_to', 'options'),
    Output('month_to', 'value'),
    Input('year_to', 'value')
)
def update_month_to_options(selected_year):
    lastYearOption_to = []
    # get the latest year from string 'yyyy-mm'
    if selected_year == theLastMonth[0:4]:
        for i in range(1, int(theLastMonth[5:7])+1):
            if i//10 < 1:
                lastMonthDict = {
                    'label': monthList[i-1], 'value': '0' + str(i)}
            else:
                lastMonthDict = {'label': monthList[i-1], 'value': str(i)}
            lastYearOption_to.append(lastMonthDict)
        options = lastYearOption_to
        value = '01'  # Set value to None by default if no option is selected
    elif selected_year == theFirstMonth[0:4]:  # the year of the first month
        laterOption = monthOption[2:]  # options start from the third month
        options = laterOption
        value = '03'  # Set value to '03' if selected_year is '2017'
    else:
        options = monthOption
        value = '06'  # Set value to None by default if no option is selected

    return options, value

#　
@app.callback(
    Output('line-chart', 'figure'),
    # Output('percentage', 'figure'),
    Input('inputLabel', 'value'),
    Input('year_from', 'value'),
    Input('month_from', 'value'),
    Input('year_to', 'value'),
    Input('month_to', 'value'),
    Input('K', 'value')
)
def update_output(label, yf, mf, yt, mt, k):
    # print(k)
    # label = "loc", yf = "2017" mf = "01" yt = "2017" mt = "03" k = [1, 10]
    # begin: the begining month
    begin = (yf+'-'+mf)
    # end: the ending month
    end = (yt+'-'+mt)
    # select dtm by the label we input than get the specific period data
    dtm0 = dtm.loc[dtm['label'] == label].loc[:, begin:end]
    # add a colomn named keywords
    dtm0['keywords'] = dtm.loc[dtm['label'] == label]['keywords']
    # add a column which is the total sum for the keyword in the period
    dtm0['period'] = dtm0.loc[:, begin:end].sum(axis=1)
    # ranking dtm0 by period
    dtm0 = dtm0.sort_values(by='period',  ascending=False)

    # dtm1 is a dtm for the keyword percentage per month
    dtm1 = dtm0.loc[:, begin:end]

    for col in dtm1.columns:
        # break
        col_sum = dtm1[col].sum()  # Calculate the sum of the column
        # percentage <- value = value/ sum of the column
        dtm1[col] = (dtm1[col] / col_sum)*100

    # select the keyword in the k interval
    selected_keywords = dtm0['keywords'][k[0]-1:k[1]]
    data = []
    data1 = []
    for keyword in selected_keywords:
        # break
        # dtm0.columns[:-2] all column names expect the last two
        y_values = dtm0[dtm0['keywords'] == keyword][list(
            dtm0.columns[:-2])].values.flatten().tolist()

        y_values1 = dtm1[dtm0['keywords'] == keyword][list(
            dtm1.columns)].values.flatten().tolist()
        data.append(
            {'x': list(dtm0.columns[:-2]), 'y': y_values, 'type': 'line', 'name': keyword})
        data1.append(
            {'x': list(dtm1.columns), 'y': y_values1, 'type': 'line', 'name': keyword})
    layout = {
        'title': 'Line Chart of Keyword Counts per Month',
        'xaxis': {'title': 'Months', 'tickformat': "%b\n%Y"},
        'yaxis': {'title': 'Counts'},

    }
    layout1 = {
        'title': 'Line Chart of Keyword percentage per Month',
        'xaxis': {'title': 'Months', 'tickformat': "%b\n%Y"},
        'yaxis': {'title': '%'},
    }
    figure0 = {'data': data, 'layout': layout}
    figure1 = {'data': data1, 'layout': layout1}
    return (figure0)

app.run_server(debug=True, use_reloader=True)
# app.run_server("0.0.0.0",port=8089, debug=True)