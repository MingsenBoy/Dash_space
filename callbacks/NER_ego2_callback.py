import pandas as pd
import re
import numpy as np
import gc
from dash.dependencies import Input, Output, State
from utils.data_loader import *

# In[]
# 測試用
# Z = "OroraTech" Unit = "Sentence" type = 'co-occurrence' stratum1_num = 7 threshold = 0.5 input_filter = ["不篩選"]
# pattern:Z = 'HELSINKI'  threshold = 0.3 stratum2_num = 0
# Z = "Carnegie_Mellon_University"   Unit = "Document" input_filter = []
# Z = "3D printing"
# Unit = "Sentence"  type = 'correlation' total_nodes_num = 10  threshold = 0.4 input_filter = "不篩選"


# In[]
n_clicks_counter = 0
X2 = RAW_DOC.copy()
senlabel2 = senlabel.copy()
raw_S2 = RAW_SEN.copy()
raw_D2 = RAW_DOC.copy()

def filter_node_result(pattern, type):

    global origin_key_dict_pd2
    global CO_sen2
    global CO_doc2
    global CR_doc2
    global CR_sen2
    global XX_Doc2
    global XX_Sent2
    

    corr_list = []
    corr_list2 = []
    
    origin_key_dict_pd2 = origin_key_dict_pd.copy()
    CR_doc2 = CR_DOC.copy()
    CR_sen2 = CR_SEN.copy()
    CO_doc2 = CO_DOC.copy()
    CO_sen2 = CO_SEN.copy()
    XX_Sent2 = DTM_SEN.copy()
    XX_Doc2 = DTM_DOC.copy()
    
    if pattern in origin_key_dict_pd['keywords'].tolist():
        # print("keywords in Dict")
        
        return origin_key_dict_pd, DTM_SEN, DTM_DOC, CR_DOC, CR_SEN, CO_SEN, CO_DOC
        
    else:
        
        pattern_rf = rf'\b{re.escape(pattern)}\b'


        # 計算新字的freq，新字計算結果放進字典
        pattern_count = (RAW_DOC["ner_doc"].apply(
            lambda x: len(re.findall(pattern_rf, x, flags=re.IGNORECASE)))).sum()
        append_row = {'keywords': pattern, 'label': "term",
                      'word_label_count': pattern_count, 'freq': pattern_count}
        origin_key_dict_pd2.loc[len(origin_key_dict_pd2)] = append_row
        
        # 計算新字在sent和doc出現次數，放入矩陣中
        XX_Sent2[pattern] = RAW_SEN['ner_sen'].str.contains(
            pattern_rf, case=False).astype(int)
        XX_Doc2[pattern] = RAW_DOC['ner_doc'].str.contains(
            pattern_rf, case=False).astype(int)
        
        pattern_xDocu = RAW_DOC['ner_doc'].str.contains(
            pattern_rf, case=False).astype(int)
        pattern_stm = RAW_SEN['ner_sen'].str.contains(
            pattern_rf, case=False).astype(int)
        
    
        # CO
        XX_Sent2_non_zero_index = XX_Sent2.loc[XX_Sent2[pattern] != 0].index
        XX_Sent2_non_zero_index_df = XX_Sent2.loc[XX_Sent2_non_zero_index]
        XX_Sent2_column_sums = (XX_Sent2_non_zero_index_df.sum()).tolist()
        del XX_Sent2_column_sums[-1]
        keyword3 = XX_Sent2_column_sums
        CO_sen2[pattern] = keyword3
        keyword3.append(0)
        CO_sen2.loc[CO_sen2.shape[0]] = keyword3
        del keyword3
        del XX_Sent2_column_sums
        gc.collect()
        
        XX_Doc2_non_zero_index = XX_Doc2.loc[XX_Doc2[pattern] != 0].index
        XX_Doc2_non_zero_index_df = XX_Doc2.loc[XX_Doc2_non_zero_index]
        XX_Doc2_column_sums = (XX_Doc2_non_zero_index_df.sum()).tolist()
        del XX_Doc2_column_sums[-1]
        keyword4 = XX_Doc2_column_sums
        CO_doc2[pattern] = keyword4
        keyword4.append(0)
        CO_doc2.loc[CO_doc2.shape[0]] = keyword4
        del keyword4
        del XX_Doc2_column_sums
        gc.collect()
        
        # CR
        # 計算新字與其他字的相關性(sent)
        for i in range(len(DTM_DOC.columns)):
            corr_list.append(pattern_xDocu.corr(
                DTM_DOC[DTM_DOC.columns.tolist()[i]], method='pearson'))
        
        # 取出計算結果把結果與cr$xDocu合併
        keyword = corr_list
        CR_doc2[pattern] = keyword
        keyword.append(1)
        CR_doc2.loc[CR_doc2.shape[0]] = keyword
        del keyword
        del corr_list
        gc.collect()
        
        # 計算新字與其他字的相關性(sent)
        for j in range(len(DTM_SEN.columns)):
            corr_list2.append(pattern_stm.corr(
                DTM_SEN[DTM_SEN.columns.tolist()[j]], method='pearson'))
        
        # 取出計算結果把結果與cr$Sent合併
        keyword2 = corr_list2
        CR_sen2[pattern] = keyword2
        keyword2.append(1)
        CR_sen2.loc[CR_sen2.shape[0]] = keyword2
        del keyword2
        del corr_list2
        gc.collect()


        return origin_key_dict_pd2, XX_Sent2, XX_Doc2, CR_doc2, CR_sen2, CO_sen2, CO_doc2
   
# In[]
# 計算edge寬度用
def calculate_edge_width(x, Min, Max):
    if Max - Min > 0:
        return (x - Min) / (Max - Min)
    else:
        return x

# 網路圖函數 Unit：計算單位(句、篇) Z：中心節點字 type：計算單位（CO或CR）total_nodes_num:網路圖節點數量 threshold:計算閥值 input_filter:篩選遮罩參數


def get_element_modify(Unit, Z, type, stratum1_num, stratum2_num,threshold, input_filter):

    all_node_list = []
    node_size_list = []
    uni_all_node_list = []
    # 按照條件篩選資料
    if type == 'correlation':
        if Unit == "Document":
            input_data = CR_doc2
        elif Unit == "Sentence":
            input_data = CR_sen2
        
        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd2['label']) if label not in input_filter]
            input_filter_list = [
                index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd2['label'], origin_key_dict_pd2['keywords']))
                if keyword == Z or label not in input_filter
            ]
            v = [(index, input_data.loc[index, Z])
                 for index in input_filter_list]

        else:
            v = input_data[Z].tolist()
            v = list(enumerate(v))

        # 過濾掉 v 中的 NaN 值
        v = [(i, val) for i, val in v if not np.isnan(val)]

        v = sorted(v, key=lambda x: float(x[1]), reverse=True)  # 降序排列
        v_index = [i for i, _ in v][:stratum1_num]  # 取出前K個索引

        # 逐個節點判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        for z_index in v_index:

            if isinstance(input_filter, list):
                # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd2['label']) if label not in input_filter]
                input_filter_list = [
                    index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd2['label'], origin_key_dict_pd2['keywords']))
                    if keyword == Z or label not in input_filter
                ]
                v = [(index, input_data.loc[index, input_data.columns.tolist()[z_index]])
                     for index in input_filter_list]

            else:
                v = input_data[input_data.columns.tolist()[z_index]].tolist()
                v = list(enumerate(v))

            # 過濾掉 v 中的 NaN 值
            v = [(i, val) for i, val in v if not np.isnan(val)]
            v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
            if stratum2_num > 0:
                v_index_2nd_level = [i for i, _ in v][:stratum2_num]  # 取出前K個索引 correspond keyword number of second-level
            else:
                v_index_2nd_level = []  # consider the no second-level scenario 
            # The * operator is used to unpack the elements of v_index_2nd_level so that they are added individually to all_node_list.
            all_node_list.extend([z_index, *v_index_2nd_level])
            # v_index = [i for i, _ in v][:total_nodes_num]  # 取出前K個索引

            # all_node_list.extend(v_index)
        del v
        gc.collect()

        uni_all_node_list = pd.unique(all_node_list).tolist()

        col_index = [((input_data.columns).tolist())[i]
                     for i in uni_all_node_list]  # 獲取對應的欄位名
        x = input_data.loc[uni_all_node_list, col_index]  # v_index,col_index取值
        x.columns = uni_all_node_list

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
        del x_updated
        gc.collect()

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

        # 新增['from_name','to_name','id']的欄位，透過索引映射到對應值
        melted_df_thres['from_name'] = melted_df_thres['from'].map(
            dict(zip(uni_all_node_list, col_index)))
        melted_df_thres['to_name'] = melted_df_thres['to'].map(
            dict(zip(uni_all_node_list, col_index)))
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
            node_size_list.append(float(XX_Sent2[node].sum()))
                #origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['freq'].to_string().split()[1]))
        # for node in nodes_list:
        #     node_size_list.append(float(
        #         origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['freq'].to_string().split()[1]))

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

        input_data = pd.DataFrame(input_data)
        input_data.columns = (choose_data.columns.tolist())
        np.fill_diagonal(input_data.values, 0)

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd2['label']) if label not in input_filter]
            input_filter_list = [
                index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd2['label'], origin_key_dict_pd2['keywords']))
                if keyword == Z or label not in input_filter
            ]
            v = [(index, choose_data.loc[index, Z])
                 for index in input_filter_list]

        else:
            v = choose_data[Z].tolist()
            v = list(enumerate(v))
        # 過濾掉 v 中的 NaN 值
        v = [(i, val) for i, val in v if not np.isnan(val)]
        v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
        v_index = [i for i, _ in v][:stratum1_num]  # 取出前K個索引

        # 逐個節點判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        for z_index in v_index:
            if isinstance(input_filter, list):
                # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd2['label']) if label not in input_filter]
                input_filter_list = [
                    index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd2['label'], origin_key_dict_pd2['keywords']))
                    if keyword == Z or label not in input_filter
                ]
                v = [(index, input_data.loc[index, input_data.columns.tolist()[z_index]])
                     for index in input_filter_list]

            else:
                v = input_data[input_data.columns.tolist()[z_index]].tolist()
                v = list(enumerate(v))

            v = [(i, val) for i, val in v if not np.isnan(val)]
            v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
            # v_index = [i for i, _ in v][:total_nodes_num]  # 取出前K個索引

            # all_node_list.extend(v_index)
            if stratum2_num > 0:
                v_index_2nd_level = [i for i, _ in v][:stratum2_num]  # correspond keyword number for the second level
            else:
                v_index_2nd_level = []  # No second-level branches

            all_node_list.extend([z_index, *v_index_2nd_level])
        del v
        gc.collect()

        uni_all_node_list = pd.unique(all_node_list).tolist()

        col_index = [((input_data.columns).tolist())[i]
                     for i in uni_all_node_list]  # 獲取對應的欄位名
        x = input_data.loc[uni_all_node_list, col_index]  # v_index,col_index取值
        x.columns = uni_all_node_list

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
        del x_updated
        gc.collect()

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

        # 新增['from_name','to_name','id']的欄位，透過索引映射到對應值
        melted_df_thres['from_name'] = melted_df_thres['from'].map(
            dict(zip(uni_all_node_list, col_index)))
        melted_df_thres['to_name'] = melted_df_thres['to'].map(
            dict(zip(uni_all_node_list, col_index)))
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
        # 字典對應節點的freq值
        for node in nodes_list:
            node_size_list.append(float(XX_Sent2[node].sum()))
                #origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['freq'].to_string().split()[1]))
        # for node in nodes_list:
        #     node_size_list.append(float(
        #         origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['freq'].to_string().split()[1]))

        # 用以計算節點大小
        size_total = sum(node_size_list)

    # group:節點字類別 title:網路圖tooltip shape:節點形狀 size:節點大小
    nodes = [
        {
            'id': node,
            'label': node,
            'group': origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['label'].to_string().split()[1],
            'title': node + ":({},{})".format(origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['label'].to_string().split()[1], int(XX_Sent2[node].sum())),
            #'title': node + ":({},{})".format(origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['label'].to_string().split()[1], origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]),
            'shape': 'dot',
            'size': 15 + (float(XX_Sent2[node].sum()/size_total))*125,
            #'size': 15 + (int(origin_key_dict_pd2[origin_key_dict_pd2['keywords'] == node]['freq'].to_string().split()[1])/size_total)*100,
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
            "stratum1_num": stratum1_num,
            "stratum2_num": stratum2_num,
            # "total_nodes_num": total_nodes_num,
            "threshold": threshold
            }

    data = {'nodes': nodes,
            'edges': edges,
            'info': info
            }

    return data

# 取得node值在原始資料的值和顏色填充
def node_recation(Unit, data):

    colored_sen_list = []
    
    k = data[0]  # 所點擊的node值

    v = XX_Sent2[(XX_Sent2[k] >= 1)] # 取關鍵詞矩陣，矩陣中值大於1的索引
    v = v[k]
    v = v.index.tolist()
    index = RAW_SEN.loc[v]  # 透過索引取值

    # 資料合併
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
    merged_df = pd.merge(merged_df, X2, on='doc_id', how='left')

    # 資料按時間排序
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df.sort_values(
        by='date', ascending=False).reset_index(drop=True)
    

    if len(merged_df) > 1000:
        merged_df = merged_df[:999]
    
    # print(merged_df)
        
    #merged_df = merged_df[merged_df['sen_kw_list'] == k].reset_index(drop=True)
    
    if k not in origin_key_dict_pd['keywords'].tolist():
        label = "term"
        list_index = (CLASS_LIST.index(label))

        for index, row in merged_df.iterrows():
            ner_sen = row['ner_sen']
            token_matches = re.finditer(
                rf'\b{re.escape(k)}\b', ner_sen, flags=re.IGNORECASE)
            offset = 0
            for match in token_matches:
                start = match.start() + offset
                end = match.end() + offset
                text_color = COLOUR[list_index]
                span_len = len(f"<span style='color: {text_color};'></span>")
                ner_sen = "{}<span style='color: {};'>{}</span>{}".format(
                    ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])
                offset += span_len
            colored_sen_list.append(ner_sen)
        
    else:
        merged_df = merged_df[merged_df['sen_kw_list'] == k].reset_index(drop=True)

        for index, row in merged_df.iterrows():

            label = row['label']
            list_index = (CLASS_LIST.index(label))
            ner_sen = row["ner_sen"]
            
            # 若一個句子具有兩個以上相同實體(token)
            if len(merged_df[(merged_df['doc_id'] == int(row["doc_id"])) & (merged_df['sen_id'] == int(row["sen_id"]))]) > 1:
                # 將此句子的實體進行標註
                duplicate_token_df = merged_df[(merged_df['doc_id'] == int(
                    row["doc_id"])) & (merged_df['sen_id'] == int(row["sen_id"]))]
                duplicate_token_df = duplicate_token_df.sort_values(by='start')
                offset = 0

                # print(f"SEN:{int(row['sen_id'])}, start with{int(row['start'])}")
                # print(duplicate_token_df['start'])
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

# In[]
# edge點擊事件

def edge_recation(Unit, data):

    colored_sen_list = []
    # from,to token
    from_to_token = data[0].split("_")
    from_token = from_to_token[0]
    to_token = from_to_token[1]

    if Unit == "Sentence":
        # from_token,to_token取關鍵詞矩陣
        token_df = XX_Sent2[[from_token, to_token]]
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[(token_df[from_token] >= 1)
                            & (token_df[to_token] >= 1)]
        index = RAW_SEN.loc[token_df.index.tolist()]  # index取值

        # 欄位合併，篩出非關鍵字
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X2, on='doc_id', how='left')
        merged_df2 = merged_df2[
            (merged_df2['ner_sen'].str.contains(from_token, case=False)) |
            (merged_df2['ner_sen'].str.contains(to_token, case=False))
            ]

        #if to_token.lower() == from_token.lower() | from_token.lower() == to_token.lower():
            
            

        # 兩關鍵字實體數量不同，數量大的或為基準
        if len(merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(
                drop=True)) > len(merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)):
            
            main_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            token = to_token
            token2 = from_token
            
            # label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == to_token]['label'].values[0]
            # label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == from_token]['label'].values[0]
        else:
            main_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            token = from_token
            token2 = to_token
            
        if token not in origin_key_dict_pd['keywords'].tolist():
            # token是自定義關鍵字（數量較多）
            label = "term"
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token2]['label'].values[0]

            for index, row in main_df.iterrows():

                list_index = (CLASS_LIST.index(label2))
                ner_sen = row["ner_sen"]

                # 先標記一般的關鍵字
                
                # 若一個句子具有兩個以上相同實體(token)
                if len(sub_df[(sub_df['doc_id'] == int(row["doc_id"])) & (sub_df['sen_id'] == int(row["sen_id"]))]) > 1:
                    # 將此句子的實體進行標註
                    duplicate_token_df = merged_df[(merged_df['doc_id'] == int(
                        row["doc_id"])) & (merged_df['sen_id'] == int(row["sen_id"]))]
                    duplicate_token_df = duplicate_token_df.sort_values(by='start')
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
                    
                # 在標記自定義關鍵字
                token_matches = re.finditer(
                    rf'\b{re.escape(token)}\b', colored_text, flags=re.IGNORECASE)
                # 潛在問題：如果自定義關鍵字包括了<span style='color: {};'>{}</span>裡面的字詞可能會出現問題
                list_index2 = CLASS_LIST.index(label)

                offset = 0
                for match in token_matches:
                    start = match.start() + offset
                    end = match.end() + offset
                    text_color = COLOUR[list_index2]
                    span_len = len(f"<span style='color: {text_color};'></span>")
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], text_color, colored_text[start:end], colored_text[end:])
                    offset += span_len

                colored_sen_list.append(colored_text)
            
        elif token2 not in origin_key_dict_pd['keywords'].tolist():
            # 有自定義關鍵字（數量較少）
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token]['label'].values[0]
            label2 = "term"
            for index, row in main_df.iterrows():

                list_index = (CLASS_LIST.index(label))
                ner_sen = row["ner_sen"]

                # 先標記一般的關鍵字
                
                # 若一個句子具有兩個以上相同實體(token)
                if len(main_df[(main_df['doc_id'] == int(row["doc_id"])) & (main_df['sen_id'] == int(row["sen_id"]))]) > 1:
                    # 將此句子的實體進行標註
                    duplicate_token_df = merged_df[(merged_df['doc_id'] == int(
                        row["doc_id"])) & (merged_df['sen_id'] == int(row["sen_id"]))]
                    duplicate_token_df = duplicate_token_df.sort_values(by='start')
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
                    
                # 在標記自定義關鍵字
                token_matches = re.finditer(
                    rf'\b{re.escape(token2)}\b', colored_text, flags=re.IGNORECASE)
                # 潛在問題：如果自定義關鍵字包括了<span style='color: {};'>{}</span>裡面的字詞可能會出現問題

                list_index2 = CLASS_LIST.index(label2)
                offset = 0
                for match in token_matches:
                    start = match.start() + offset
                    end = match.end() + offset
                    text_color = COLOUR[list_index2]
                    span_len = len(f"<span style='color: {text_color};'></span>")
                    colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                        colored_text[:start], text_color, colored_text[start:end], colored_text[end:])
                    offset += span_len

                colored_sen_list.append(colored_text)
        else:
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token]['label'].values[0]
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token2]['label'].values[0]

            # 資料型態轉為datatime後，降序排列
            main_df['date'] = pd.to_datetime(main_df['date']).dt.date
            main_df = main_df.sort_values(
                by='date', ascending=False).reset_index(drop=True)

            for index, row in main_df.iterrows():

                ner_sen = row["ner_sen"]
                offset = 0
                
                list_index = (CLASS_LIST.index(label))
                list_index2 = (CLASS_LIST.index(label2))

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
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'], how='left')
        merged_df2 = pd.merge(merged_df2, RAW_DOC, on='doc_id', how='left')
        merged_df2 = merged_df2[merged_df2['doc_id'].isin(doc_df.index)]

        # 兩關鍵字實體數量不同，數量大的或為基準
        if len(merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(
                drop=True)) > len(merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)):
            main_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            token = to_token
            token2 = from_token
        else:
            main_df = merged_df2[merged_df2['sen_kw_list'] == from_token].reset_index(drop=True)
            sub_df = merged_df2[merged_df2['sen_kw_list'] == to_token].reset_index(drop=True)
            token = from_token
            token2 = to_token
            
        if token not in origin_key_dict_pd['keywords'].tolist():
            label = "term"
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token2]['label'].values[0]
        elif token2 not in origin_key_dict_pd['keywords'].tolist():
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token]['label'].values[0]
            label2 = "term"
        else:
            label = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token]['label'].values[0]
            label2 = origin_key_dict_pd[origin_key_dict_pd['keywords'] == token2]['label'].values[0]

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
    
    # 避免datatable元件過載
    if len(main_df) > 1000:
        main_df = main_df[:999]
    
    return main_df, from_token, to_token

def register_callback(app):
    
    @app.callback(
        Output("net", 'data'),
        Input('RadioItems_SenorDoc', 'value'),
        [Input('submit_button', 'n_clicks')],
        [State('input_text', 'value')],
        Input('RadioItems_CRorCO', 'value'),
        Input("stratum1_num", "value"),
        Input("stratum2_num", "value"),
        Input('threshold_slide', 'value'),
        Input('dropdown_choose_filter', 'value'),

    )
    def update_elements(Unit, n_clicks, input_text, type, stratum1_num, stratum2_num,  threshold, input_filter):
        global n_clicks_counter
        global Unit_2
        global input_text_2
        global total_nodes_num_2
        global type_2
        global threshold_2
        global input_filter_2

        if n_clicks >= n_clicks_counter:
            # print(n_clicks_counter)
            # print(n_clicks)
            n_clicks_counter = n_clicks + 1
            Unit_2 = Unit
            input_text_2 = input_text
            # total_nodes_num_2 = total_nodes_num
            type_2 = type
            threshold_2 = threshold
            input_filter_2 = input_filter
            origin_key_dict_pd2, XX_Sent2, XX_Doc2, CR_doc2, CR_sen2, CO_sen2, CR_sen2 = filter_node_result(
                input_text_2, type_2)

            return get_element_modify(Unit_2, input_text_2, type_2, stratum1_num, stratum2_num, threshold_2, input_filter_2)

        elif n_clicks < n_clicks_counter:

            return get_element_modify(Unit, input_text, type, stratum1_num, stratum2_num,  threshold, input_filter)


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

    # Datatable更新函數
    @app.callback(
        Output('table', 'data'),
        Output('doc_number', 'data'),
        Input('RadioItems_SenorDoc', 'value'),
        Input('net', 'selection')
    )
    def update_elements(Unit, selection):
        global merged_df
        res = []

        if len(selection['nodes']) != 0:
            # 將node對應資料映射到datatable
            merged_df, token = node_recation(
                Unit, selection['nodes'])
            for i, j, k, l in zip(merged_df['date'], merged_df['doc_id'], merged_df['colored_sen'], merged_df['url']):
                res.append({'Date': i, 'id': "<a href= " + l + " target='_blank'>" +
                        str(j) + "</a>", 'Recent': k})
        elif len(selection['edges']) != 0:
            # 將edge對應資料映射到datatable
            merged_df2, from_token, to_token = edge_recation(
                Unit, selection['edges'])
            for i, j, k, l in zip(merged_df2['date'], merged_df2['doc_id'], merged_df2['colored_sen'], merged_df2['url']):
                res.append({'Date': i, 'id': "<a href= " + l + " target='_blank'>" +
                        str(j) + "</a>", 'Recent': k})
        else:
            pass
        # Count the number of unique doc_id in the output list
        unique_doc_ids = len(set(item['id'] for item in res))

        # Count the total number of doc_id in the output list
        total_doc_ids = len([item['id'] for item in res])

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
            color_doc = RAW_DOC[RAW_DOC['doc_id'] == doc_id]
            color_doc = pd.merge(color_doc, doclabel, on='doc_id', how='left')
            colored_text = color_doc['ner_doc'].values[0]
            # 點擊的是node時顯示對應的text area
            if len(selection['nodes']) != 0:
                # 如果選取的node不再orgin_key_dict表示是自定義關鍵字，用正規表達式標色
                if(selection['nodes'][0] not in origin_key_dict_pd['keywords'].tolist()):
                    label = "term"
                    list_index = (CLASS_LIST.index(label))
                    k = selection['nodes'][0]
                    token_matches = re.finditer(
                        rf'\b{re.escape(k)}\b', colored_text, flags=re.IGNORECASE)
                    for match in token_matches:
                        start = match.start() + offset
                        end = match.end() + offset
                        text_color = COLOUR[list_index]
                        span_len = len(f"<span style='color: {text_color};'></span>")
                        colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                            colored_text[:start], text_color, colored_text[start:end], colored_text[end:])
                        offset += span_len
                else:
                # selection is a dictionary containing 'nodes' and 'edges' 
                # if the node is selected, 'nodes' will demonstrate the selected keyword 
                # while the 'edges' display the branches all related to the chosen keyword.               
                    color_doc_id_df = color_doc[color_doc['doc_kw_list']==selection['nodes'][0]]
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
            # 點擊edges顯示對應的text area
            elif len(selection['edges']) != 0:
                # if the edge is selected, nothing will demonstrate in 'nodes'
                # while the 'edges' display the selected edge, which connects two keywords by _
                node_list = selection['edges'][0].split('_')
                defined_keywords = [item for item in node_list if item not in origin_key_dict_pd['keywords'].tolist()]
                # 把edge兩邊的關鍵字的label_table資料找出來，用start進行排序後標記顏色
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

                # 若有自定義關鍵字，則另外進行顏色標記
                if defined_keywords:
                    label = "term"
                    list_index = (CLASS_LIST.index(label))
                    offset = 0
                    token_matches = re.finditer(
                        rf'\b{re.escape(defined_keywords[0])}\b', colored_text, flags=re.IGNORECASE)
                    for match in token_matches:
                        start = match.start() + offset
                        end = match.end() + offset
                        text_color = COLOUR[list_index]
                        span_len = len(f"<span style='color: {text_color};'></span>")
                        colored_text = "{}<span style='color: {};'>{}</span>{}".format(
                            colored_text[:start], text_color, colored_text[start:end], colored_text[end:])
                        offset += span_len

                res = [{'article': colored_text}]

            return res

    # 調整文章字體大小
    @app.callback(
        Output('textarea_table', 'style_table'),
        Input('textarea-font-size-slider', 'value')
    )
    def update_textarea_font_size(selected_font_size):
        style_table = {'height': '300px',
                    'overflowY': 'auto', 'fontSize': selected_font_size}
        return style_table