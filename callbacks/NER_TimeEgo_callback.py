# 切換 class 下拉式選單
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto  # pip install dash-cytoscape
import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np
import re
import gc
from utils.data_loader import *
from scipy import sparse

# In[]
n_clicks_counter = 0  # 按鈕參數

# 抓取最新和最舊的資料時間
latest_date = RAW_DOC['date'].max()
start_date = RAW_DOC['date'].min()


# 無人機修改
latest_date_object = datetime.strptime(latest_date, '%Y-%m-%d')
start_date_object = datetime.strptime(start_date, '%Y-%m-%d')
# latest_date_object = datetime.strptime(latest_date, '%Y-%m-%d %H:%M:%S')
# start_date_object = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')

def calculate_edge_width(x, Min, Max):
    if Max - Min > 0:
        return (x - Min) / (Max - Min)
    else:
        return x

# 網路圖函數 Unit：計算單位(句、篇) Z：中心節點字 type：計算單位（CO或CR）total_nodes_num:網路圖節點數量 threshold:計算閥值 input_filter:篩選遮罩參數
def node_recation(start_date, end_date, Unit, data, type, total_nodes_num, threshold):

    colored_sen_list = []

    k = data[0]  # 所點擊的node值
    v = XX_Sent2[(XX_Sent2[k] >= 1)] # 取關鍵詞矩陣，矩陣中值大於1的索引
    v = v[k]
    v = v.index.tolist()
    index = RAW_SEN.loc[v]  # 透過索引取值
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    # 資料合併、篩選
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
    merged_df = pd.merge(merged_df, RAW_DOC, on='doc_id', how='left')
    merged_df = merged_df[merged_df['sen_kw_list'] == k].reset_index(drop=True)

    # 資料按時間排序
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df[merged_df['date']>= start_date]
    merged_df = merged_df[merged_df['date']<= end_date]
    merged_df = merged_df.sort_values(
        by='date', ascending=False).reset_index(drop=True)

    if len(merged_df) > 1000:
        merged_df = merged_df[:999]

    for index, row in merged_df.iterrows():

        label = row["label"]
        list_index = (CLASS_LIST.index(label))
        ner_sen = row["ner_sen"]
        
        # 若一個句子具有兩個以上相同實體(token)
        if len(merged_df[(merged_df['doc_id'] == int(row["doc_id"])) & (merged_df['sen_id'] == int(row["sen_id"]))]) > 1:
            # 將此句子的實體進行標註
            duplicate_token_df = merged_df[(merged_df['doc_id'] == int(
                row["doc_id"])) & (merged_df['sen_id'] == int(row["sen_id"]))]
            offset = 0
            for index, row in duplicate_token_df.iterrows():
                start = int(row["start"]) + offset
                end = int(row["end"]) + offset
                text_color = COLOUR[list_index]

                span_len = len(f"<span style='color: {text_color};'></span>")
                if offset == 0:
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])
                else:
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], text_color, colored_text[start:end], colored_text[end:])
                offset += span_len 
        else:
            start = int(row["start"])
            end = int(row["end"])
            text_color = COLOUR[list_index]
            colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])

        colored_sen_list.append(colored_text)

    merged_df["colored_sen"] = colored_sen_list
    #刪除重複['doc_id', 'sen_id']的row
    merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)

    return merged_df, k


def edge_recation(start_date, end_date, Unit, data, type, total_nodes_num, threshold):
    
    colored_sen_list = []
    from_to_token = data[0].split("_")
    from_token = from_to_token[0]
    to_token = from_to_token[1]
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    if Unit == "Sentence":
        token_df = XX_Sent2[[from_token, to_token]]  # from_token,to_token取關鍵詞矩陣
        token_df['total'] = token_df.sum(axis=1)
        token_df = token_df[(token_df[from_token] >= 1)
                            & (token_df[to_token] >= 1)]
        index = RAW_SEN.loc[token_df.index.tolist()]  # index取值

        # 欄位合併
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, RAW_DOC, on='doc_id', how='left')
        merged_df2 = merged_df2[(merged_df2['sen_kw_list'] == from_token) | (
            merged_df2['sen_kw_list'] == to_token)]

        # 兩關鍵字實體數量不同，數量大的或為基準
        if len(merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(
                drop=True)) > len(merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)):
            
            main_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            token = to_token
            token2 = from_token
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == to_token]['label'].values[0]
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == from_token]['label'].values[0]
        else:
            main_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            token = from_token
            token2 = to_token
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == from_token]['label'].values[0]
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == to_token]['label'].values[0]

        # 資料型態轉為datatime後，降序排列
        main_df['date'] = pd.to_datetime(main_df['date']).dt.date
        main_df = main_df[main_df['date']>= start_date]
        main_df = main_df[main_df['date']<= end_date]
        main_df = main_df.sort_values(
            by='date', ascending=False).reset_index(drop=True)

        for index, row in main_df.iterrows():

            ner_sen = row["ner_sen"]
            list_index = (CLASS_LIST.index(label))
            list_index2 = (CLASS_LIST.index(label2))
            offset = 0

            all_entities = pd.concat([main_df, sub_df])
            all_entities = all_entities[(all_entities['doc_id'] == int(row["doc_id"])) & (all_entities['sen_id'] == int(row["sen_id"]))]
            all_entities = all_entities.sort_values(by='start')  # 根據 start 進行排序

            for _, ent_row in all_entities.iterrows():
                start = int(ent_row["start"]) + offset
                end = int(ent_row["end"]) + offset
                if ent_row['sen_kw_list'] == token:
                    text_color = COLOUR[list_index]
                else:
                    text_color = COLOUR[list_index2]
                
                span_len = len(f"<span style='color: {text_color};'></span>")
                if offset == 0:
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])
                else:
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], text_color, colored_text[start:end], colored_text[end:])

                offset += span_len  # 更新總偏移量

            colored_sen_list.append(colored_text)

    else:
        token_df = XX_Sent2[[from_token, to_token]]  # from_token,to_token取關鍵詞矩陣
        token_df = token_df[(token_df[from_token] >= 1)
                            | (token_df[to_token] >= 1)]
        index = RAW_SEN.loc[token_df.index.tolist()]  # index取值
        index = index[index.duplicated('doc_id', keep=False)]  # 僅保留同篇文章資料
        doc_df = XX_Doc2[(XX_Doc2[from_token] >= 1) & (XX_Doc2[to_token] >= 1)] # 找出包含兩個的document

        # 欄位合併
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, RAW_DOC, on='doc_id', how='left')
        merged_df2 = merged_df2[merged_df2['doc_id'].isin(doc_df.index)]

        # 兩關鍵字實體數量不同，數量大的或為基準
        if len(merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(
                drop=True)) > len(merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)):
            main_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            token = to_token
            token2 = from_token
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == to_token]['label'].values[0]
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == from_token]['label'].values[0]
        else:
            main_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            token = from_token
            token2 = to_token
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == from_token]['label'].values[0]
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == to_token]['label'].values[0]

        # 資料型態轉為datatime後，降序排列
        main_df['date'] = pd.to_datetime(main_df['date']).dt.date
        main_df = main_df.sort_values(
            by='date', ascending=False).reset_index(drop=True)
               
        
        for index, row in main_df.iterrows():

            ner_sen = row["ner_sen"]
            list_index = (CLASS_LIST.index(label))
            list_index2 = (CLASS_LIST.index(label2))
            offset = 0

            all_entities = pd.concat([main_df, sub_df])
            all_entities = all_entities[(all_entities['doc_id'] == int(row["doc_id"])) & (all_entities['sen_id'] == int(row["sen_id"]))]
            all_entities = all_entities.sort_values(by='start')  # 根據 start 進行排序


            for _, ent_row in all_entities.iterrows():
                start = int(ent_row["start"]) + offset
                end = int(ent_row["end"]) + offset
                if ent_row['sen_kw_list'] == token:
                    text_color = COLOUR[list_index]
                else:
                    text_color = COLOUR[list_index2]
                
                span_len = len(f"<span style='color: {text_color};'></span>")
                if offset == 0:
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])
                else:
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], text_color, colored_text[start:end], colored_text[end:])

                offset += span_len  # 更新總偏移量
            
            colored_sen_list.append(colored_text)

    main_df["colored_sen"] = colored_sen_list
    #刪除重複['doc_id', 'sen_id']的row
    main_df = main_df.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
    
    # 防止datatable元件過載
    if len(main_df) > 1000:
        main_df = main_df[:999]

    return main_df, from_token, to_token

def get_element_modify(Unit, Z, type, total_nodes_num, threshold, input_filter):

    node_size_list = []
    input_filter_list = []
    if total_nodes_num > 21:
        total_nodes_num = 21
    # 按照條件篩選資料
    if type == 'correlation':
        if Unit == "Document":
            input_data = CR_doc2
        elif Unit == "Sentence":
            input_data = CR_sen2

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]# 從字典挑選非遮罩的詞索引
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

        x_values = x.values  # 獲取x數據部分，轉換為numpy數組
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
        del x_updated
        gc.collect()

        # 閥值計算
        value_list = melted_df["Value"].tolist()
        
        if len(value_list) == 0:
            percentile = None 
        else:
            percentile = np.percentile(value_list, (threshold * 100))  # 根據value_list算出閥值

        melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(
            drop=True)  # 符合threshold的value
        melted_df_thres["Value"] = np.sqrt(
            melted_df_thres['Value'].astype('float'))  # 取平方根值
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

        for node in nodes_list:
            node_size_list.append(int(XX_Doc2[node].sum()))

        # 用以計算節點大小
        size_total = sum(node_size_list)

    # 按照條件篩選資料
    elif type == 'co-occurrence':
        if Unit == "Document":
            input_data = CO_doc2
            choose_data = CR_doc2
        elif Unit == "Sentence":
            input_data = CO_sen2
            choose_data = CR_sen2

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]# 從字典挑選非遮罩的詞索引
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
        melted_df_thres["Value"] = np.sqrt(
            melted_df_thres['Value'].astype('float'))  # 取平方根值
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
        nodes_list = list(set(nodes_list))

        for node in nodes_list:
            node_size_list.append(int(XX_Doc2[node].sum()))

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
            'size': 15 + (int(XX_Doc2[node].sum())/size_total)*100,
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

def sparse_corr(A):
    # N 是樣本數，即行數
    N = A.shape[0]
    
    # 計算協方差矩陣 C
    # A.T*A 是矩陣A的轉置與其自身的乘積
    # sum(A) 是矩陣A中每一列的和，sum(A).T*sum(A) 是將其轉置後再與自己相乘
    # 這一步是計算樣本間的協方差
    C = ((A.T * A - (sum(A).T * sum(A) / N)) / (N - 1)).todense()
    
    # 計算變異數（對角線元素），V 是變異數的平方根
    # np.diag(C) 提取矩陣 C 的對角線元素，這些元素是每個特徵的變異數
    V = np.sqrt(np.mat(np.diag(C)).T * np.mat(np.diag(C)))
    
    # 計算相關係數矩陣，避免除以零，加上一個小的常數 1e-119
    COR = np.divide(C, V + 1e-119)
    
    # 返回相關係數矩陣
    return COR

def time_range_data(start_date, end_date):

    global X2
    global raw_S2
    global CR_doc2
    global CO_doc2
    global CR_sen2
    global CO_sen2
    global XX_Doc2
    global XX_Sent2

    # 資料長度修整
    XX_Sent_length = len(DTM_SEN)
    XX_Doc_length = len(DTM_DOC)

    X2 = RAW_DOC.iloc[:XX_Doc_length]
    raw_S2 = RAW_SEN.iloc[:XX_Sent_length]

    # 依照X2['date']抓取時間區間資料
    X2['date'] = pd.to_datetime(X2['date'])
    Doc_date_index = X2[(X2['date'] >= start_date) & (X2['date'] <= end_date)]
    doc_id_list = Doc_date_index['doc_id']
    XX_Doc2 = DTM_DOC.loc[Doc_date_index.index]
    doc_columns = XX_Doc2.columns
    CSR_Doc2 = sparse.csr_matrix(XX_Doc2.values)

    # DocCR
    CR_doc2 = sparse_corr(CSR_Doc2)
    CR_doc2 = pd.DataFrame(CR_doc2, columns=doc_columns)
    np.fill_diagonal(CR_doc2.values, 1)
    CR_doc2 = CR_doc2.fillna(0)

    # DocCO
    CO_doc2 = np.dot(CSR_Doc2.T, CSR_Doc2)
    # DocCO = pd.DataFrame(DocCO.values, columns=XX_Doc_data.columns, index=XX_Doc_data.columns)
    CO_doc2 = pd.DataFrame.sparse.from_spmatrix(CO_doc2, columns=doc_columns)
    np.fill_diagonal(CO_doc2.values, 0)
    CO_doc2 = CO_doc2.fillna(0)

    Sent_date_index = raw_S2[raw_S2['doc_id'].isin(doc_id_list)]
    XX_Sent2 = DTM_SEN.loc[Sent_date_index.index]
    sent_columns = XX_Sent2.columns
    CSR_Sent2 = sparse.csr_matrix(XX_Sent2.values)

    # Sen_CR
    CR_sen2 = sparse_corr(CSR_Sent2)
    CR_sen2 = pd.DataFrame(CR_sen2, columns=sent_columns)
    CR_sen2.reset_index(inplace=True, drop=True)
    np.fill_diagonal(CR_sen2.values, 1)
    CR_sen2 = CR_sen2.fillna(0)

    # Sen_CO
    CO_sen2 = np.dot(CSR_Sent2.T, CSR_Sent2)
    # DocCO = pd.DataFrame(DocCO.values, columns=XX_Doc_data.columns, index=XX_Doc_data.columns)
    CO_sen2 = pd.DataFrame.sparse.from_spmatrix(CO_sen2, columns=sent_columns)
    np.fill_diagonal(CO_sen2.values, 0)
    CO_sen2 = CO_sen2.fillna(0)

    return X2, raw_S2, CR_doc2, CO_doc2, CR_sen2, CO_sen2, XX_Doc2, XX_Sent2




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
        Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date'),
    )
    def update_elements(Unit, selection, total_nodes_num, type, threshold, start_date, end_date):
        global merged_df
        res = []

        if len(selection['nodes']) != 0:
            # print(selection)
            # 將node對應資料映射到datatable
            merged_df, token = node_recation(
                start_date, end_date, Unit, selection['nodes'], total_nodes_num, type, threshold)
            for i, j, k, l in zip(merged_df['date'], merged_df['doc_id'], merged_df['colored_sen'], merged_df['url']):
                res.append({'Date': i, 'id': "<a href= " + l + " target='_blank'>" +
                        str(j) + "</a>", 'Recent': k})
        elif len(selection['edges']) != 0:
            # print(selection)
            # 將edge對應資料映射到datatable
            merged_df2, from_token, to_token = edge_recation(
                start_date, end_date, Unit, selection['edges'], total_nodes_num, type, threshold)
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
        Input('table', 'active_cell'),
        Input('net', 'selection')
    )
    def myfun(active_cell, selection):
        # print(active_cell)
        res = []
        offset = 0
        if active_cell == None:
            return res
        else:
            pattern = re.compile(r'<a.*?>(\d+)</a>')
            match_doc_id = re.match(pattern, active_cell['row_id']).group(1)
            doc_id = int(match_doc_id)
            # print(doc_id)
            color_doc = RAW_DOC[RAW_DOC['doc_id'] == doc_id]
            color_doc = pd.merge(color_doc, doclabel, on='doc_id', how='left')
            colored_text = color_doc['ner_doc'].values[0]
            if len(selection['nodes']) != 0:
                # selection is a dictionary containing 'nodes' and 'edges' 
                # if the node is selected, 'nodes' will demonstrate the selected keyword 
                # while the 'edges' display the branches all related to the chosen keyword.               
                color_doc_id_df = color_doc[color_doc['doc_kw_list']==selection['nodes'][0]]
                color_doc_id_df = color_doc_id_df.sort_values(by='start')
                print(color_doc_id_df)

                for index, row in color_doc_id_df.iterrows():
                    start = int(row["start"]) + offset
                    end = int(row["end"]) + offset
                    color_index = CLASS_LIST.index(row["label"])
                    color = COLOUR[color_index]
                    span_len = len(f"<span style='color: {color};'></span>")
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], color, colored_text[start:end], colored_text[end:])
                    offset += span_len
                res = [{'article': colored_text}]
            elif len(selection['edges']) != 0:
                # if the edge is selected, nothing will demonstrate in 'nodes'
                # while the 'edges' display the selected edge, which connects two keywords by _
                node_list = selection['edges'][0].split('_')
                color_doc_id_df = color_doc[color_doc['doc_kw_list'].isin(node_list)]
                color_doc_id_df = color_doc_id_df.sort_values(by='start')
                for index, row in color_doc_id_df.iterrows():
                    start = int(row["start"]) + offset
                    end = int(row["end"]) + offset
                    color_index = CLASS_LIST.index(row["label"])
                    color = COLOUR[color_index]
                    span_len = len(f"<span style='color: {color};'></span>")
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], color, colored_text[start:end], colored_text[end:])
                    offset += span_len
                res = [{'article': colored_text}]

            return res

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


    # 調整文章字體大小
    @app.callback(
        Output('textarea_table', 'style_table'),
        Input('textarea-font-size-slider', 'value')
    )
    def update_textarea_font_size(selected_font_size):
        style_table = {'height': '300px',
                    'overflowY': 'auto', 'fontSize': selected_font_size}
        return style_table
    
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
        [Input('submit_button', 'n_clicks')],
        Input('RadioItems_SenorDoc', 'value'),
        Input("dropdown_choose_name", 'value'),
        Input("total_nodes_num", "value"),
        Input('RadioItems_CRorCO', 'value'),
        Input('threshold_slide', 'value'),
        Input('dropdown_choose_filter', 'value'),
        Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date')
    )
    def update_elements(n_clicks, Unit, center_node, total_nodes_num, type, threshold, input_filter, start_date, end_date):

        global n_clicks_counter
        global Unit_2
        global total_nodes_num_2
        global type_2
        global threshold_2
        global input_filter_2
        global start_date2
        global end_date2

        if n_clicks >= n_clicks_counter:

            n_clicks_counter = n_clicks + 1
            X2, raw_S2, CR_doc2, CO_doc2, CR_sen2, CO_sen2, XX_Doc2, XX_Sent2 = time_range_data(
                start_date, end_date)
            return get_element_modify(Unit, center_node, type, total_nodes_num, threshold, input_filter)

        elif n_clicks < n_clicks_counter:
            
            return get_element_modify(Unit, center_node, type, total_nodes_num, threshold, input_filter)


