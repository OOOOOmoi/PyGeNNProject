#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 vip-ext-scale 扫描的群体放电率，重点观察 V 层。"""
import os, json
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

# vip=1.0 无后缀(即 ext0.02), 其余有 _vipextX 后缀
cases = [("1.0", "ext0.02"), ("0.5", "ext0.02_vipext0.5"), ("0.3", "ext0.02_vipext0.3"),
         ("0.2", "ext0.02_vipext0.2"), ("0.1", "ext0.02_vipext0.1"), ("0.05", "ext0.02_vipext0.05")]

print("vipext  " + "  ".join(f"{p:>6}" for p in pops))
for tag, sub in cases:
    sdir = os.path.join(BASE, f"spike_0.5s_AreaNum1_{sub}_GPU1")
    if not os.path.exists(sdir):
        print(f"{tag:>7}  (missing: {sub})")
        continue
    rates = [rate(sdir, p) for p in pops]
    print(f"{tag:>7}  " + "  ".join(f"{r:6.2f}" for r in rates))
