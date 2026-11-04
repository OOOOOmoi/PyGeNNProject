#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""高分辨率二维网格扫描：解析 400 组合目录名，输出群体放电率矩阵。"""
import os, json, glob
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']

def rate(sdir, p):
    f = os.path.join(sdir, "V1", f"V1_{p}_spikes.csv")
    if not os.path.exists(f):
        return 0.0
    a = np.loadtxt(f, delimiter=',')
    if a.size == 0:
        return 0.0
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a.shape[0] / nn[p] / DUR_S

# 0.05 步长，20 个值
VALS = [round(0.05*i, 2) for i in range(1, 21)]  # 0.05 ... 1.00

def find_dir(v, s):
    name = "spike_0.5s_AreaNum1_ext0.02"
    if v != 1.0:
        name += f"_vipext{v:g}"
    if s != 1.0:
        name += f"_sstext{s:g}"
    return os.path.join(BASE, name + "_GPU1")

# 校验
missing = []
for v in VALS:
    for s in VALS:
        dd = find_dir(v, s)
        if not os.path.exists(dd):
            missing.append((v, s))
print(f"总组合: {len(VALS)*len(VALS)}, 缺失: {len(missing)}")
if missing:
    for m in missing[:20]:
        print("  missing:", m)

# 保存矩阵到 npy 供后续绘图/分析
matrices = {}
for p in pops:
    mat = np.array([[rate(find_dir(v, s), p) for s in VALS] for v in VALS])
    matrices[p] = mat

np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/hires_matrices.npz",
         **{p: matrices[p] for p in pops})
print("saved hires_matrices.npz")

# 输出关键群体的矩阵（部分展示）
print("\n值列表:", VALS)
