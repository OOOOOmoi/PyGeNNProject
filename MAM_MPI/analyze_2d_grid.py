#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VIP × SST 二维网格扫描：解析目录名，输出各群体放电率矩阵。"""
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

VIP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
SST = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

def find_dir(v, s):
    """按 getModelName 规则解析目录名（默认值 1.0 无后缀）。"""
    name = "spike_0.5s_AreaNum1_ext0.02"
    if v != 1.0:
        name += f"_vipext{v:g}"
    if s != 1.0:
        name += f"_sstext{s:g}"
    return os.path.join(BASE, name + "_GPU1")

# 校验所有目录是否存在
missing = []
for v in VIP:
    for s in SST:
        dd = find_dir(v, s)
        if not os.path.exists(dd):
            missing.append((v, s, os.path.basename(dd)))
if missing:
    print("MISSING dirs:")
    for m in missing:
        print("  ", m)
else:
    print("all 49 dirs found")

# 输出每个群体的 7x7 矩阵
print("\n# 群体放电率矩阵 (行=vipext, 列=sstext)")
for p in pops:
    print(f"\n## {p}")
    print("vip\\sst " + "  ".join(f"{s:>7}" for s in SST))
    for v in VIP:
        row = [rate(find_dir(v, s), p) for s in SST]
        print(f"{v:>7}  " + "  ".join(f"{r:7.2f}" for r in row))
