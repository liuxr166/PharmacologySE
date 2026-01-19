# -*- coding: utf-8 -*-
"""
@Time    : 2026-01-03
@Auth    :
@File    : tools.py
@IDE     : PyCharm
@Edition : 001
@Describe:
"""
import math
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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
