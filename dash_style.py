
from dash import html


STYLE_FOR_TITLE = {
    'font-size': '36px',
    'textAlign': 'center',
    'backgroundColor': '#daf5ed',
    'margin': '0px',
    'font-weight': 'bold',
    'padding': '5px'
}

STYLE_FOR_SUBTITLE = {
    'font-size': '24px',
    'textAlign': 'center',
    'backgroundColor': '#f2efe4',
    'padding': '3px',
    'margin': '0px',
}


def get_ledgend(COLOUR, LEGEND):
    # 設定ledgend
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
    
    return legend