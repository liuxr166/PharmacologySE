# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/11
@Auth    :
@File    : get_drug_target.py
@IDE     : PyCharm
@Edition : 001
@Describe:
"""
import json
import xmltodict
import pandas as pd

# pd.set_option('display.max_columns', None)  # 显示完整的列
# pd.set_option('display.max_rows', None)  # 显示完整的行

# # 读取 XML 文件并转换为 JSON
# xml = open("full_database.xml", encoding="utf-8").read()
# json_file = "data.json"
# convertJson = xmltodict.parse(xml, encoding="utf-8")
# jsonStr = json.dumps(convertJson, indent=4)
#
# with open(json_file, 'w+', encoding="utf-8") as f:
#     f.write(jsonStr)

# 读取 JSON 数据
with open('data.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

drugs = data['drugbank']['drug']

drug_list = list()
drug_df = pd.DataFrame()

for drug in drugs:
    drug_dict = dict()
    target_list = list()
    enzyme_list = list()
    pubchemId = None
    try:
        external_ids = drug['external-identifiers']['external-identifier']
        if isinstance(external_ids, dict):
            if external_ids['resource'] == 'PubChem Compound':
                pubchemId = external_ids['identifier']

        if isinstance(external_ids, list):
            for external_id in external_ids:
                if external_id['resource'] == 'PubChem Compound':
                    pubchemId = external_id['identifier']
    except:
        pubchemId = None
    try:
        drug_id = drug['drugbank-id'][0]['#text']
    except:
        drug_id = drug['drugbank-id']['#text']
    drug_name = drug['name']
    try:
        targets = drug['targets']['target']
    except:
        targets = drug['targets']
    try:
        enzymes = drug['enzymes']['enzyme']
    except:
        enzymes = drug['enzymes']

    if isinstance(targets, dict):
        try:
            uniprot_id = targets['polypeptide']['@id']
        except:
            uniprot_id = targets['id']
        target_list.append(uniprot_id)

    if isinstance(targets, list):
        for target in targets:
            try:
                uniprot_id = target['polypeptide']['@id']
            except:
                uniprot_id = target['id']
            target_list.append(uniprot_id)

    if isinstance(enzymes, dict):
        try:
            uniprot_id = enzymes['polypeptide']['@id']
        except:
            uniprot_id = enzymes['id']
        enzyme_list.append(uniprot_id)

    if isinstance(enzymes, list):
        for enzyme in enzymes:
            try:
                uniprot_id = enzyme['polypeptide']['@id']
            except:
                uniprot_id = enzyme['id']
            enzyme_list.append(uniprot_id)

    drug_dict['drugId'] = drug_id
    drug_dict['pubchemId'] = pubchemId
    drug_dict['drugName'] = drug_name
    drug_dict['target'] = target_list
    drug_dict['enzyme'] = enzyme_list

    drug_list.append(drug_dict)

drug_df = pd.DataFrame(drug_list)
drug_df.to_pickle('items.pkl')

# data = pd.read_pickle('items.pkl')
# print(data.loc[data.drugId == 'DB01016']['target'])  # object
# print(data.loc[data.drugId == 'DB01016']['target'].values[0])  # list

# print(drug_df.iloc[:50, :])
# print(drug_df.loc[drug_df.pubchemId == '2244']['target'])  # object
# for i in drug_df.loc[drug_df.pubchemId == '2244']['target']:
#     print(i)
