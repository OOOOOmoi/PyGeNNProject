#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""二维联合扫描结果：输出 V4/S4 等群体放电率矩阵（修正 sstext=1.0 无后缀）。"""
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

VIP = ["0.30","0.35","0.40","0.45"]
SST = ["0.3","0.5","0.7","1.0"]
focus = ["V4","S4","V5","V6","E4","E23","S23","P4"]

def find_dir(v, s):
    # sstext=1.0 无后缀
    if s == "1.0":
        cand = os.path.join(BASE, f"spike_0.5s_AreaNum1_ext0.02_vipext{float(v):g}_GPU1")
        return cand if os.path.exists(cand) else None
    cand = os.path.join(BASE, f"spike_0.5s_AreaNum1_ext0.02_vipext{float(v):g}_sstext{float(s):g}_GPU1")
    return cand if os.path.exists(cand) else None

for p in focus:
    print(f"\n===== {p} 放电率矩阵 (行=vipext, 列=sstext) =====")
    print("vipext\\sst " + "  ".join(f"{s:>8}" for s in SST))
    for v in VIP:
        row = []
        for s in SST:
            sdir = find_dir(v, s)
            r = rate(sdir, p) if sdir else float('nan')
            row.append(f"{r:8.2f}")
        print(f"{v:>10}  " + "  ".join(row))
