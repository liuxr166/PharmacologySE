# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/21
@Auth    :
@File    : get_drug_feat.py
@IDE     : PyCharm
@Edition : 001
@Describe: 
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# pd.set_option('display.max_columns', None)  # 显示完整的列
np.set_printoptions(threshold=np.inf)


def feat2vec(drug_feats, name_feats):
    list_feat = list()
    certain_feat = np.array(drug_feats[name_feats]).tolist()  # ["P30556|P05412","P28223|P46098|……"]
    for feat in certain_feat:
        feat = str(feat)
        for ft in feat.split('|'):
            if ft not in list_feat:
                list_feat.append(ft)  # 获得全部种类的靶点/酶
    feature_matrix = np.zeros((len(certain_feat), len(list_feat)), dtype=float)
    matrix_feat = pd.DataFrame(feature_matrix, columns=list_feat)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(certain_feat)):
        for ft in str(drug_feats[name_feats].iloc[i]).split('|'):
            matrix_feat[ft].iloc[i] = 1

    # print(matrix_feat)
    matrix_feat = np.array(matrix_feat)
    return matrix_feat  # 返回shape(drug_num, feature_num)的矩阵


drug_feats = pd.read_csv('df_drugs.csv')
matrix_smile = feat2vec(drug_feats, name_feats='smile')  # (1569, 881)
matrix_target = feat2vec(drug_feats, name_feats='target')  # (1569, 1562)
matrix_enzyme = feat2vec(drug_feats, name_feats='enzyme')  # (1569, 318)
# print(matrix_smile.shape, matrix_target.shape, matrix_enzyme.shape)

drug_pairs = pd.read_table('ddi_names.txt', sep='\t', header=None)
# print(drug_pairs.iloc[:, 2].values)
pairs_feats = list()
# print(len(np.unique(drug_pairs[2].values)))  # 81 drugs
for index, row in drug_pairs.iloc[:, :2].iterrows():  # 读取drug_pairs的每一行，并从drug_feats中获取每行每个元素的feats
    drug1 = row[0].split('::')[1]
    drug2 = row[1].split('::')[1]
    # print(drug1, drug2)  # 1337, 1510
    drug1_index = drug_feats[drug_feats.drugId == drug1].index.tolist()[0]
    drug2_index = drug_feats[drug_feats.drugId == drug2].index.tolist()[0]
    drug1_feat = np.hstack((matrix_smile[drug1_index], matrix_target[drug1_index], matrix_enzyme[drug1_index]))  # (2761,)
    # print(drug1_feat)
    drug2_feat = np.hstack((matrix_smile[drug2_index], matrix_target[drug2_index], matrix_enzyme[drug2_index]))
    drug_feat = np.hstack((drug1_feat, drug2_feat))
    pairs_feats.append(drug_feat)  # (172426, 5522)
    # break


# df_data = pd.DataFrame({
#     'data': pairs_feats,
#     'label': drug_pairs.iloc[:, 2]
# })
# df_data.to_pickle('df_data.pkl')
# print('finished!')

def cos_sim(array1, array2, pair_num):
    # 参数验证
    if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
        raise ValueError("Inputs should be numpy arrays")

    if not isinstance(pair_num, int) or pair_num <= 0:
        raise ValueError("pair_num should be a positive integer")

    # 计算余弦相似度
    cos_sim = cosine_similarity(array1, array2)

    # 获取每个元素最相似的前 pair_num 个元素的索引
    index = np.argsort(-cos_sim, axis=1)[:, :pair_num]

    return index.tolist()


pairs_feats = np.array(pairs_feats)
print(cos_sim(pairs_feats[:50000], pairs_feats[:50000], 2))
