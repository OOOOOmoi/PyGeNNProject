#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""细粒度统计 vip-ext-scale 扫描的群体放电率，定位去抑制突变边界。"""
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

# 关注群体
focus = ["V23","V4","V5","V6","S4","S23","S5","S6","E4","E23","P4","P23"]

# 直接从目录名解析 vipext 值（按 float 值排序）
dirs = glob.glob(os.path.join(BASE, "spike_0.5s_AreaNum1_ext0.02_vipext*_GPU1"))
def parse_v(d):
    name = os.path.basename(d)
    return float(name.split("vipext")[1].split("_GPU1")[0])

dirs_sorted = sorted(dirs, key=parse_v)

print("vipext  " + "  ".join(f"{p:>6}" for p in focus))
for sdir in dirs_sorted:
    v = parse_v(sdir)
    rates = [rate(sdir, p) for p in focus]
    print(f"{v:>7.3f}  " + "  ".join(f"{r:6.2f}" for r in rates))
