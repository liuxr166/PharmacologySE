# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/8
@Auth    :
@File    : get_drug_target.py
@IDE     : PyCharm
@Edition : 001
@Describe:
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from bilstm_encoder import data_processing

# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_columns', None)  # 显示完整的列
# pd.set_option('display.max_rows', None)  # 显示完整的行

conn = sqlite3.connect("./event.db")

df_drug = pd.read_sql('select * from drug;', conn)
# print(df_drug['smile'].values)
# print(df_drug.columns)
# df = df_drug.sort_values(by="id")
# print(df)
extraction = pd.read_sql('select * from extraction;', conn)
# print(extraction.columns)
mechanism = extraction['mechanism']
action = extraction['action']
drugA = extraction['drugA']
drugB = extraction['drugB']
feature_list = ["smile", "target", "enzyme"]


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    # Transfrom the interaction event to number 将相互作用转换为数字

    # 拼接mechanism和action
    d_event = list()
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    # 得到每个相互作用出现的次数
    count = dict()
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)  # 得到相互作用的个数，并返回值

    # 字典d_label用于存储相互作用的数字表示。键是list1中的元素的键，值是该元素在list1中的索引
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 将字典count按value从大到小排序，存储在list1中
    d_label = dict()
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector = list() 创建shape(len(df_drug['name']), 0)的空数组
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec = feature_vector(i, df_drug)  # smiles(572, 583) target(572, 1162) enzyme(572, 202)
        print(tempvec.shape)
        vector = np.hstack((vector, tempvec))  # 所有drug的每种特征矩阵tempvec水平拼接到vector后
    # Transfrom the drug ID to feature vector

    # 字典d_feature键是药物名，值是特征向量
    d_feature = dict()
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = list()
    new_label = list()
    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))  # 两个drug特征向量水平拼接
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])  # drug对对应的label

    new_feature = np.array(new_feature)  # 323539*....
    new_label = np.array(new_label)  # 323539

    return new_feature, new_label, event_num  # (37264, 3894) (37264,) 65


def feature_vector(feature_name, df):
    # def Jaccard(matrix):
    #     matrix = np.mat(matrix)
    #
    #     numerator = matrix * matrix.T
    #
    #     denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    #
    #     return numerator / denominator
    all_feature = list()
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features  # 获得全部种类的靶点/酶
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = pd.DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    # sim_matrix = np.array(Jaccard(df_feature))

    # print(feature_name + " len is:" + str(len(df_feature[0])))
    return df_feature  # 返回shape(drug_num,feature_num)的矩阵


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


new_feature, new_label, event_num = prepare(df_drug, feature_list, mechanism, action, drugA, drugB)
print('data has been generated!')
new_feature = np.array(new_feature)[:10000]  # 37264
print('data arrayed!')
song = data_processing(new_feature, input_dim=2, embedding_dim=64, hidden_dim=128, output_dim=64, num_layers=2, dropout=0.1, batch_size=100)
print('song\t', song)  # 大概需要20小时跑出结果
# print(type(new_feature[0][0]))  # type <class 'numpy.float64'>
# print(cos_sim(new_feature[:100], new_feature[:100], 2))
