from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import re
import gc
from summarize import generate_summary # 生成文章摘要
from utils.data_loader import *


# In[]
# 測試用
# Z = "Carnegie_Mellon_University"
# Z = "3D printing"  input_filter = ["com", "rocket", "loc"] input_filter = ["term"]
# Unit = "Sentence"  type = 'correlation' total_nodes_num = 8  threshold = 0.5 input_filter = "com"
# input_filter = "不篩選"
# In[]
# 計算edge寬度用


def calculate_edge_width(x, Min, Max):
    if Max - Min > 0:
        return (x - Min) / (Max - Min)
    else:
        return x

# 網路圖函數 Unit：計算單位(句、篇) Z：中心節點字 type：計算單位（CO或CR）total_nodes_num:網路圖節點數量 threshold:計算閥值 input_filter:篩選遮罩參數


def get_element_modify(Unit, Z, type, total_nodes_num, threshold, input_filter):

    node_size_list = []
    input_filter_list = []
    # print(input_filter)
    if total_nodes_num > 21:
        total_nodes_num = 21
    # 按照條件篩選資料
    if type == 'correlation':
        if Unit == "Document":
            input_data = CR_DOC
        elif Unit == "Sentence":
            input_data = CR_SEN

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]
            input_filter_list = [
                index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd['label'], origin_key_dict_pd['keywords']))
                if keyword == Z or label not in input_filter
            ]
            v = [(index, input_data.loc[index, Z])
                 for index in input_filter_list]

        else:
            v = input_data[Z].tolist()
            v = list(enumerate(v))

        v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
        v_index = [i for i, _ in v][:total_nodes_num]  # 取出前K個索引
        col_index = [((input_data.columns).tolist())[i]
                     for i in v_index]  # 獲取對應的欄位名
        x = input_data.loc[v_index, col_index]  # v_index,col_index取值
        x.columns = v_index
        del v
        gc.collect()

        x_values = x.values  # 獲取x的數據部分，轉換為numpy數組
        # 獲取下三角部分的boolean *x_values.shape:使用x_values數組的形狀來確定矩陣的行數和列數 dtype:設定矩陣資料型態 k:True或False比例
        lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)
        x_values[lower_triangle] = 0  # 將下三角部分（True）的元素設置為0
        # 將更新後的numpy數組重新轉換為DataFrame
        x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)
        del x
        gc.collect()

        melted_df = x_updated.stack().reset_index()  # 轉成對應關係
        melted_df.columns = ['from', 'to', 'Value']  # 欄位命名
        melted_df = melted_df[melted_df['Value']
                              > 0].reset_index(drop=True)  # 找大於0的值

        # 按['from', 'to']排序，刪除重複值
        melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
        melted_df = melted_df.drop_duplicates(
            subset=['from', 'to']).reset_index(drop=True)

        # 閥值計算
        value_list = melted_df["Value"].tolist()
        if len(value_list) == 0:
            percentile = None
        else:
            percentile = np.percentile(value_list, (threshold * 100))  # 根據value_list算出閥值

        melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(
            drop=True)  # 符合threshold的value
        melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])  # 取平方根值
        del melted_df
        gc.collect()

        # 新增['from_name','to_name','id']的欄位，透過索引映射到對應值
        melted_df_thres['from_name'] = melted_df_thres['from'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['to_name'] = melted_df_thres['to'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['id'] = melted_df_thres['from_name'].astype(
            str) + "_" + melted_df_thres['to_name'].astype(str)

        # edge的寬度計算
        Min, Max = melted_df_thres['Value'].min(
        ), melted_df_thres['Value'].max()
        melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(
            lambda x: calculate_edge_width(x, Min, Max))

        nodes_list = melted_df_thres['from_name'].tolist(
        ) + melted_df_thres['to_name'].tolist()
        nodes_list = list(set(nodes_list))

        # 字典對應節點的freq值
        for node in nodes_list:
            node_size_list.append(int(
                origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]))

        # 用以計算節點大小
        size_total = sum(node_size_list)

    # 按照條件篩選資料
    elif type == 'co-occurrence':
        if Unit == "Document":
            input_data = CO_DOC
            choose_data = CR_DOC
        elif Unit == "Sentence":
            input_data = CO_SEN
            choose_data = CR_SEN

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]
            input_filter_list = [
                index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd['label'], origin_key_dict_pd['keywords']))
                if keyword == Z or label not in input_filter
            ]
            v = [(index, choose_data.loc[index, Z])
                 for index in input_filter_list]
        else:
            v = choose_data[Z].tolist()
            v = list(enumerate(v))

        v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
        v_index = [i for i, _ in v][:total_nodes_num]  # 取出前K個索引
        col_index = [((input_data.columns).tolist())[i]
                     for i in v_index]  # 獲取對應的欄位名
        x = input_data.loc[v_index, col_index]  # v_index,col_index取值
        x.columns = v_index
        del v
        gc.collect()

        x_values = x.values  # 獲取x的數據部分，轉換為numpy數組
        # 獲取下三角部分的boolean *x_values.shape:使用x_values數組的形狀來確定矩陣的行數和列數 dtype:設定矩陣資料型態 k:True或False比例
        lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)
        x_values[lower_triangle] = 0  # 將下三角部分（包括對角線）的元素設置為0
        # 將更新後的numpy數組重新轉換為DataFrame
        x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)
        del x
        gc.collect()

        melted_df = x_updated.stack().reset_index()  # 轉成對應關係
        melted_df.columns = ['from', 'to', 'Value']  # 欄位命名
        melted_df = melted_df[melted_df['Value']
                              > 0].reset_index(drop=True)  # 找大於0的值

        # 按['from', 'to']排序，刪除重複值
        melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
        melted_df = melted_df.drop_duplicates(
            subset=['from', 'to']).reset_index(drop=True)

        # 閥值計算
        value_list = melted_df["Value"].tolist()
        
        if len(value_list) == 0:
            percentile = None
        else:
            percentile = np.percentile(value_list, (threshold * 100))  # 根據value_list算出閥值

        melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(
            drop=True)  # 取符合threshold的value
        melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])  # 取平方根值
        del melted_df
        gc.collect()

        # 新增['from_name','to_name','id']的欄位，值為透過索引映射到對應值
        melted_df_thres['from_name'] = melted_df_thres['from'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['to_name'] = melted_df_thres['to'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['id'] = melted_df_thres['from_name'].astype(
            str) + "_" + melted_df_thres['to_name'].astype(str)

        # edge的寬度計算
        Min, Max = melted_df_thres['Value'].min(
        ), melted_df_thres['Value'].max()
        melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(
            lambda x: calculate_edge_width(x, Min, Max))

        nodes_list = melted_df_thres['from_name'].tolist(
        ) + melted_df_thres['to_name'].tolist()
        nodes_list = list(set(nodes_list))  # 刪除重複值

        # 字典對應節點的freq值
        for node in nodes_list:
            node_size_list.append(int(
                origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]))

        # 用以計算節點大小
        size_total = sum(node_size_list)

    # group:節點字類別 title:網路圖tooltip shape:節點形狀 size:節點大小
    nodes = [
        {
            'id': node,
            'label': node,
            'group': origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1],
            'title': node + ":({},{})".format(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1], origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]),
            'shape': 'dot',
            'size': 15 + (int(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1])/size_total)*100,
        }
        for node in nodes_list
    ]

    # width:邊寬度 title:網路圖tooltip
    edges = [
        {
            'id': row['from_name']+'_'+row['to_name'],
            'from': row['from_name'],
            'to': row['to_name'],
            'classes': type,  # cor or coo
            'weight': row['Value'],
            'width':row['edge_width']*6,
            'title': row['from_name']+'_'+row['to_name']
        }
        for idx, row in melted_df_thres[(melted_df_thres['from_name'].isin(nodes_list) &
                                         melted_df_thres['to_name'].isin(nodes_list))].iterrows()
    ]

    info = {"Unit": str(Unit),
            "type": str(type),
            "total_nodes_num": total_nodes_num,
            "threshold": threshold
            }

    data = {'nodes': nodes,
            'edges': edges,
            'info': info
            }

    return data

# In[]
def node_recation(Unit, data, type, total_nodes_num, threshold):

    colored_sen_list = []

    k = data[0]  # 所點擊的node值
    v = DTM_SEN[k]  # 取關鍵詞矩陣
    v = np.where(v == 1)[0]  # 矩陣中值為1的索引
    v = v.tolist()
    index = RAW_SEN.loc[v]  # 透過索引取值

    # 資料合併
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
    merged_df = pd.merge(merged_df, RAW_DOC, on='doc_id', how='left')
    merged_df = merged_df.drop_duplicates(
        subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)

    # 資料按時間排序
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df.sort_values(
        by='date', ascending=False).reset_index(drop=True)

    if len(merged_df) > 1000:
        merged_df = merged_df[:999]
        
    merged_df = merged_df[merged_df['sen_kw_list'] == k].reset_index(drop=True)

    for index, row in merged_df.iterrows():

        label = row["label"]
        start = int(row["start"])
        end = int(row["end"])
        ner_sen = row["ner_sen"]

        list_index = (CLASS_LIST.index(label))
        text_color = COLOUR[list_index]
        colored_text = "{}<span style='color: {};'>{}</span>{}".format(
            ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])

        colored_sen_list.append(colored_text)

    merged_df["colored_sen"] = colored_sen_list


    return merged_df, k
# In[]
# 測試用
# data = ['ABL_UK']
# from_token = "3D printing"
# to_token = "ArianeGroup"
# In[]

def edge_recation(Unit, data, type, total_nodes_num, threshold):

    colored_sen_list = []
    # from,to token
    from_to_token = data[0].split("_")
    from_token = from_to_token[0]
    to_token = from_to_token[1]

    if Unit == "Sentence":
        token_df = DTM_SEN[[from_token, to_token]]  # from_token,to_token取關鍵詞矩陣
        token_df['total'] = token_df.sum(axis=1)
        token_df = token_df[(token_df[from_token] == 1) & (token_df[to_token] == 1)]
        index = RAW_SEN.loc[token_df.index.tolist()]  # index取值

        # 欄位合併
        from_token_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        from_token_df = pd.merge(from_token_df, RAW_DOC, on='doc_id', how='left')
        from_token_df = from_token_df[(from_token_df['sen_kw_list'] == from_token) | (from_token_df['sen_kw_list'] == to_token)]
        
        merged_df2 = from_token_df[from_token_df['sen_kw_list'] == to_token].reset_index(drop=True)
        from_token_df = from_token_df[from_token_df['sen_kw_list'] == from_token].reset_index(drop=True)

        # 資料型態轉為datatime後，降序排列
        merged_df2['date'] = pd.to_datetime(merged_df2['date']).dt.date
        merged_df2 = merged_df2.sort_values(
            by='date', ascending=False).reset_index(drop=True)
        
        from_token_df['date'] = pd.to_datetime(from_token_df['date']).dt.date
        from_token_df = from_token_df.sort_values(
            by='date', ascending=False).reset_index(drop=True)
        
        for index, row in merged_df2.iterrows():
            # if index == 0:
            #     break
            label = row["label"]
            start = int(row["start"])
            end = int(row["end"])
            
            ner_sen = row["ner_sen"]

            list_index = (CLASS_LIST.index(label))
            text_color = COLOUR[list_index]
            colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])
            
            matches = re.finditer(rf'\b{re.escape(from_token)}\b', colored_text)
            
            for match in matches:
                start2 = match.start()
                end2 = match.end()
            
            label2 = from_token_df['label'][index]
            #start2 = int(from_token_df['start'][index])
            #end2 = int(from_token_df['start'][index])
            
            list_index2 = (CLASS_LIST.index(label2))
            text_color = COLOUR[list_index2]
            colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                colored_text[:start2], text_color, colored_text[start2:end2], colored_text[end2:])
            
            colored_sen_list.append(colored_text)

    else:
        token_df = DTM_SEN[[from_token, to_token]]  # from_token,to_token取關鍵詞矩陣

        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] >= 1]
        index = RAW_SEN.loc[token_df.index.tolist()]  # index取值

        # 欄位合併
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, RAW_DOC, on='doc_id', how='left')
        merged_df2 = merged_df2[(merged_df2['sen_kw_list'] == from_token) | (merged_df2['sen_kw_list'] == to_token)]
        
        merged_df2 = merged_df2.drop_duplicates(
            subset=['doc_id', 'sen_id','sen_kw_list'], keep='first').reset_index(drop=True)
        
        duplicate_rows = merged_df2[merged_df2.duplicated(['doc_id', 'sen_id'], keep=False)].reset_index(drop=True)
        
        merged_df2 = merged_df2.drop_duplicates(
            subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
        
        # 資料型態轉為datatime後，降序排列
        duplicate_rows['date'] = pd.to_datetime(duplicate_rows['date']).dt.date
        duplicate_rows = duplicate_rows.sort_values(
            by='date', ascending=False).reset_index(drop=True)
        
        # 資料型態轉為datatime後，降序排列
        merged_df2['date'] = pd.to_datetime(merged_df2['date']).dt.date
        merged_df2 = merged_df2.sort_values(
            by='date', ascending=False).reset_index(drop=True)

        for index, row in merged_df2.iterrows():

            label = row["label"]
            start = int(row["start"])
            end = int(row["end"])
            doc_id = int(row["doc_id"])
            sen_id = int(row["sen_id"])
            ner_sen = row["ner_sen"]
            
            list_index = (CLASS_LIST.index(label))
            text_color = COLOUR[list_index]
            colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])
            
            if ((duplicate_rows['doc_id'] == doc_id) & (duplicate_rows['sen_id'] == sen_id)).any():
                
                if row["sen_kw_list"] == to_token:
                    
                    another_token_row = duplicate_rows.loc[
                        (duplicate_rows['doc_id'] == doc_id) &
                        (duplicate_rows['sen_id'] == sen_id) &
                        (duplicate_rows['sen_kw_list'] == from_token)
                    ]
                    
                else:      
                    
                    another_token_row = duplicate_rows.loc[
                        (duplicate_rows['doc_id'] == doc_id) &
                        (duplicate_rows['sen_id'] == sen_id) &
                        (duplicate_rows['sen_kw_list'] == to_token)
                    ]
                    
                matches = re.finditer(rf'\b{re.escape(from_token)}\b', colored_text)
                
                for match in matches:
                    start2 = match.start()
                    end2 = match.end()
                            
                label2 = another_token_row['label'].values[0]
                           
                list_index2 = (CLASS_LIST.index(label2))
                text_color = COLOUR[list_index2]
                colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                colored_text[:start2], text_color, colored_text[start2:end2], colored_text[end2:])

            colored_sen_list.append(colored_text)

    # 防止datatable元件過載
    if len(merged_df2) > 1000:
        merged_df2 = merged_df2[:999]

    merged_df2["colored_sen"] = colored_sen_list

    return merged_df2, from_token, to_token

def register_callback(app):
    # Datatable更新函數
    @app.callback(
        Output('table', 'data'),
        Output('doc_number', 'data'),
        Input('RadioItems_SenorDoc', 'value'),
        Input('net', 'selection'),
        Input("total_nodes_num", "value"),
        Input('RadioItems_CRorCO', 'value'),
        Input('threshold_slide', 'value'),
    )
    def update_elements(Unit, selection, total_nodes_num, type, threshold):
        global merged_df
        res = []

        if len(selection['nodes']) != 0:
            # print(selection)
            # 將node對應資料映射到datatable
            merged_df, token = node_recation(
                Unit, selection['nodes'], total_nodes_num, type, threshold)
            for i, j, k, l in zip(merged_df['date'], merged_df['doc_id'], merged_df['colored_sen'], merged_df['url']):
                res.append({'Date': i, 'id': "<a href= " + l + " target='_blank'>" +
                        str(j) + "</a>", 'Recent': k})
        elif len(selection['edges']) != 0:
            # print(selection)
            # 將edge對應資料映射到datatable
            merged_df2, from_token, to_token = edge_recation(
                Unit, selection['edges'], total_nodes_num, type, threshold)
            for i, j, k, l in zip(merged_df2['date'], merged_df2['doc_id'], merged_df2['colored_sen'], merged_df2['url']):
                res.append({'Date': i, 'id': "<a href= " + l + " target='_blank'>" +
                        str(j) + "</a>", 'Recent': k})
        else:
            pass
        # Count the number of unique doc_id in the output list
        unique_doc_ids = len(set(item['id'] for item in res))
        # print(f"Number of unique doc_id: {unique_doc_ids}")

        # Count the total number of doc_id in the output list
        total_doc_ids = len([item['id'] for item in res])
        # print(f"Total number of doc_id: {total_doc_ids}")

        # Prepare data for doc_number DataTable
        doc_number_data = [
            {'doc_id_num': unique_doc_ids, 'recent_num': total_doc_ids}]
        return res, doc_number_data

    # textarea更新函數
    @app.callback(
        Output('textarea_table', 'data'),
        Output('article-id', 'children'),
        Input('table', 'active_cell'),
        Input('net', 'selection')
    )
    def myfun(active_cell, selection):
        # print(active_cell)
        res = []
        if active_cell == None:
            return res, ""
        else:
            pattern = re.compile(r'<a.*?>(\d+)</a>')
            match_doc_id = re.match(pattern, active_cell['row_id']).group(1)
            doc_id = int(match_doc_id)
            color_doc = RAW_DOC[RAW_DOC['doc_id'] == doc_id]['ner_doc'].values[0]
            # print(color_doc)
            if len(selection['nodes']) != 0:
                # selection is a dictionary containing 'nodes' and 'edges' 
                # if the node is selected, 'nodes' will demonstrate the selected keyword 
                # while the 'edges' display the branches all related to the chosen keyword.               
                color_doc_id_df= origin_key_dict_pd[origin_key_dict_pd['keywords']==selection['nodes'][0]]  
            elif len(selection['edges']) != 0:
                # if the edge is selected, nothing will demonstrate in 'nodes'
                # while the 'edges' display the selected edge, which connects two keywords by _
                node_list = selection['edges'][0].split('_')
                color_doc_id_df = origin_key_dict_pd[origin_key_dict_pd['keywords'].isin(node_list)]
            for row in color_doc_id_df.itertuples():
                keyword = row.keywords 
                color_index = CLASS_LIST.index(row.label)
                color = COLOUR[color_index]
                color_doc = color_doc.replace(keyword, f"<span style='color: {color}'>{keyword}</span>")
            res = [{'article': color_doc}]
            return res, doc_id

    # 調整節點字體大小
    @app.callback(
        Output('net', 'options'),
        Input('nodes-font-size-slider', 'value')
    )
    def update_nodes_font_size(selected_font_size):
        options = {
            'nodes': {
                'font': {
                    'size': selected_font_size
                }
            }
        }
        return options

    # 調整文章字體大小
    @app.callback(
        Output('textarea_table', 'style_table'),
        Input('textarea-font-size-slider', 'value')
    )
    def update_textarea_font_size(selected_font_size):
        style_table = {'height': '300px',
                    'overflowY': 'auto', 'fontSize': selected_font_size}
        return style_table

    # 生成文章摘要
    @app.callback(
        Output("summary-modal", "is_open"),
        Output("summary-content", "children"),
        Output("summary-loading-output", "children"),
        Input("summary-button", "n_clicks"),
        State("summary-modal", "is_open"),
        State('textarea_table', 'data'),
        State('article-id', 'children')
    )
    def toggle_modal(n1, is_open, table_data, doc_id):
        if n1:
            if table_data:
                # Extract the content from the first cell assuming it's a single cell
                first_cell_content = table_data[0]['article']  
                summary = generate_summary(doc_id, first_cell_content)
                return not is_open, summary, ""
            else:
                return not is_open, "請選擇文章", ""
        return is_open, "None", ""
        
    # 切換 class 下拉式選單
    @app.callback(
        Output("dropdown_choose_name", 'options'),
        Input("dropdown_choose_class", "value"),
    )
    def update_elements(class_idx):  # 當dropdown_choose_class下拉選單的值發生變化時，會觸發，class_idx類別索引
        # 選擇中心詞
        options = [
            {'label': name, 'value': name}
            for name in origin_key_dict_pd[origin_key_dict_pd['label'] == CLASS_LIST[class_idx]]['keywords'].to_list()
        ]

        return options


    # 更新下拉選單
    @app.callback(
        Output("threshold_slide", 'min'),
        Output("threshold_slide", 'max'),
        Output("threshold_slide", 'marks'),
        Output("threshold_slide", 'value'),
        Input("RadioItems_CRorCO", 'value')
    )
    def update_elements(type):
        # if type == 'correlation':
        min = 0
        max = 1
        marks = {i/10: str(i/10) for i in range(11)}
        value = 0.3

        if type == 'co-occurrence':
            min = 0
            max = 1
            marks = {i/10: str(i/10) for i in range(11)}
            value = 0.3

        return min, max, marks, value

    # 當dropdown-update-layout下拉選單的值發生變化時，更新網路圖


    @app.callback(
        Output("net", 'data'),
        Input('RadioItems_SenorDoc', 'value'),
        Input("dropdown_choose_name", 'value'),
        Input("total_nodes_num", "value"),
        Input('RadioItems_CRorCO', 'value'),
        Input('threshold_slide', 'value'),
        Input('dropdown_choose_filter', 'value'),
    )
    def update_elements(Unit, center_node, total_nodes_num, type, threshold, input_filter):

        return get_element_modify(Unit, center_node, type, total_nodes_num, threshold, input_filter)