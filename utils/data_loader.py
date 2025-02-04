import pandas as pd

NER_TOPIC = "太空文集"
# NER_TOPIC = "無人機文集"
# NER_TOPIC = "無人機文集"

# DATA_SOURCE_NAME = "SpaceNews"
# DEAFAULT_VALUE = "Antares"

# LEGEND = ["company", "product", "term", "location", "organization"]
LEGEND = ["company", "rocket", "orgnization", "satellite", "term", "location",]
# 類別顏色
COLOUR = ["#8DD3C7", "#F781BF", "#BEBADA", "#FB8072", "#92D050", "#FDB462"]

Sen_Doc_list = ["Sentence", "Document"]

if(NER_TOPIC == '太空文集'):
    DATA_SOURCE_NAME = "SpaceNews"
    DEAFAULT_VALUE = "NASA"
    # 關鍵字類別 太空文集
    CLASS_LIST = ["com", "rocket", "org", "satellite", "term", "loc"]
    # 讀取資料集 太空新聞
    origin_key_dict_pd = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\entityDict_TEST.csv')
    senlabel = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\sen_label_table.csv')
    doclabel = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\doc_label_table.csv')

    DTM_DOC = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\DocDTM.csv')
    DTM_SEN = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\SenDTM.csv')   
    RAW_DOC = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\doc_raw_data.csv')
    RAW_SEN = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\sen_raw_data.csv')
    CR_DOC = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\DocCR.csv')
    CR_SEN = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\SenCR.csv')
    CO_DOC = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\DocCO.csv')
    CO_SEN = pd.read_csv(r'D:\Lab\project\spacenews\dash\data3\SenCO.csv')    
elif(NER_TOPIC == '無人機文集'):
    DATA_SOURCE_NAME = "Dronelife"
    DEAFAULT_VALUE = "Zipline"
    # 關鍵字類別 無人機
    CLASS_LIST = ["com", "product", "term", "loc", "org"]
    
    # 讀取資料集 無人機
    origin_key_dict_pd = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/entityDict.csv')
    senlabel = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/sen_label_table.csv')
    doclabel = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/doc_label_table.csv')

    DTM_DOC = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/DocDTM.csv')
    DTM_SEN = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/SenDTM.csv')   
    RAW_DOC = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/doc_raw_data.csv')
    RAW_SEN = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/sen_raw_data.csv')
    CR_DOC = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/DocCR.csv')
    CR_SEN = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/SenCR.csv')
    CO_DOC = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/DocCO.csv')
    CO_SEN = pd.read_csv('/mnt/NAS_shared/mirdc_ner/drone/SenCO.csv')
elif(NER_TOPIC == '氫能文集'):
    # 關鍵字類別 氫能
    CLASS_LIST =  ["loc", "org", "technology", "product", "proj"]
    
    CLASS_FULL_NAMES = {
        "loc": "Location",
        "org": "Organization",
        "technology": "Technology",
        "product": "Product",
        "proj": "Project"
    }

    # 讀取資料集 氫能
    # origin_key_dict_pd = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/entityDict_TEST.csv')
    # senlabel = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/sen_label_table.csv')
    # doclabel = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/doc_label_table.csv')

    # DTM_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/DocDTM.csv')
    # DTM_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/SenDTM.csv')   
    # RAW_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/doc_raw_data.csv')
    # RAW_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/sen_raw_data.csv')
    # CR_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/DocCR.csv')
    # CR_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/SenCR.csv')
    # CO_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/DocCO.csv')
    # CO_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/SenCO.csv')

    # 讀取資料集 氫能
    origin_key_dict_pd = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/entityDict_TEST.csv')
    senlabel = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/sen_label_table.csv')
    doclabel = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/doc_label_table.csv')

    DTM_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/DocDTM.csv')
    DTM_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/SenDTM.csv')   
    RAW_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/doc_raw_data.csv')
    RAW_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/sen_raw_data.csv')
    CR_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/DocCR.csv')
    CR_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/SenCR.csv')
    CO_DOC = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/DocCO.csv')
    CO_SEN = pd.read_csv('/home/mirdc/MIRDC_DASH/DASH_Hydrogen/data/hydrogen_test/SenCO.csv')



