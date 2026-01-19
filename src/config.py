# -*- coding: utf-8 -*-
"""
@Time    : 2026-01-03
@Auth    :
@File    : config.py.py
@IDE     : PyCharm
@Edition : 001
@Describe:
"""
# 模型参数
seed = 2
epo_num = 100
pair_num = 51
Att_n_heads = 4
batch_size = 128
cross_ver_tim = 5
drop_out_rating = 0.5
learn_rating = 1.0e-5
weight_decay_rate = 1.0e-5
feature_list = ["smile", "target", "enzyme"]

# 设备设置
DEVICE_TYPE = 'npu'  # 可选：'npu', 'cuda', 'cpu'
CUDA_VISIBLE_DEVICES = '0'

# 路径设置
file_path = "./results/"
