# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/20
@Auth    :
@File    : get_drug_list.py
@IDE     : PyCharm
@Edition : 001
@Describe:
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

pd.set_option('display.max_columns', None)  # 显示完整的列


def sml2fp(smile_list) -> list:
    fp_list = list()
    for sml in smile_list:
        mol = Chem.MolFromSmiles(sml)
        morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=881)
        fp_list.append(list(morgan_hashed))
    return fp_list


def _separator(input_lists, flag) -> list:
    """

    :param input_lists: 一组由子列表组成的列表
    :param flag: 0->smiles, 1->targets, 2->enzymes
    :return: 每个子列表转换为'|'分隔的字符串
    """
    output_strings = list()

    for lst in input_lists:
        if flag == 0:
            # 找到值为1的索引
            indices = [str(idx) for idx, char in enumerate(lst) if char == 1]
            # 将索引用|分隔，组成新的字符串
            new_string = '|'.join(indices)
        else:
            new_string = '|'.join(lst)
        # 将结果添加到输出列表
        output_strings.append(new_string)

    return output_strings


df_ids = pd.read_table('ddi_names.txt', sep='\t', header=None)  # 从这里找id
# 获取全部药物id
drug_ids = np.unique([df_ids[0], df_ids[1]])  # np.unique()  type <class 'numpy.ndarray'>  len 1569
list_ids = list(i.split('::')[1] for i in drug_ids)
# print(list_ids)


# 获取全部药物smiles串
df_smiles = pd.read_table('ddi_smiles.txt', sep='\t', header=None).sort_values(by=[0])  # 从这里找smiles串
index_smiles = df_smiles[df_smiles[0].isin(drug_ids)].index.tolist()
list_smiles = df_smiles.loc[index_smiles, 1].tolist()
list_smiles = _separator(sml2fp(list_smiles), flag=0)
# print(list_smiles)


# 获取全部药物target和enzyme
df_targets = pd.read_pickle('items.pkl')  # 从这里找target和enzyme
# print(df_data.loc[df_data.drugId == 'DB13925', :])
df_targets = df_targets.loc[:, ['drugId', 'target', 'enzyme']].sort_values(by=['drugId'])
index_targets = df_targets[df_targets['drugId'].isin(list_ids)].index.tolist()
# print(index_targets)
list_targets = df_targets.loc[index_targets, 'target'].tolist()
list_targets = _separator(list_targets, flag=1)
# print(list_targets)  # 存在target为''

list_enzymes = df_targets.loc[index_targets, 'enzyme'].tolist()
# print(list_enzymes)
list_enzymes = _separator(list_enzymes, flag=2)
# print(list_enzymes)  # 存在enzyme为''

df_drugs = pd.DataFrame({
    'drugId': list_ids,
    'smile': list_smiles,
    'target': list_targets,
    'enzyme': list_enzymes
})
df_drugs.to_csv('df_drugs.csv')
