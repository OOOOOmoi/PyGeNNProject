#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 inh-scale 扫描的群体放电率（不计算加权平均）。"""
import os, json
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']

inhl = ["1.0","1.5","2.0","3.0","4.0","5.0"]

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

print("inh  " + "  ".join(f"{p:>6}" for p in pops))
for inh in inhl:
    sdir = os.path.join(BASE, f"spike_0.5s_AreaNum1_ext0.02_inh{inh}_GPU1")
    if not os.path.exists(sdir):
        print(f"{inh:>5}  (missing)")
        continue
    rates = [rate(sdir, p) for p in pops]
    print(f"{inh:>5}  " + "  ".join(f"{r:6.2f}" for r in rates))
