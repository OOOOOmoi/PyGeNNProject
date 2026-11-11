#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量统计 V1 单区 ext-scale 扫描的群体放电率。"""
import os, glob, json
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']
Ntot = sum(nn.values())

exts = ["0.001","0.002","0.003","0.004","0.005","0.006","0.008","0.01","0.015","0.02","0.03","0.05"]

def get_rate(spike_dir, pop):
    f = os.path.join(spike_dir, "V1", f"V1_{pop}_spikes.csv")
    if not os.path.exists(f):
        return 0.0, 0
    a = np.loadtxt(f, delimiter=',')
    if a.size == 0:
        a = np.empty((0, 2))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = a.shape[0]
    return n / nn[pop] / DUR_S, n

# 表头
print("ext-scale  " + "  ".join(f"{p:>6}" for p in pops) + "  weighted_mean  total_spikes")
for ext in exts:
    sdir = os.path.join(BASE, f"spike_0.5s_AreaNum1_ext{ext}_GPU1")
    rates = []
    ts = 0
    for p in pops:
        r, n = get_rate(sdir, p)
        rates.append(r)
        ts += n
    wmean = sum(nn[p] * rates[i] for i, p in enumerate(pops)) / Ntot
    row = f"{ext:>10}  " + "  ".join(f"{r:6.2f}" for r in rates) + f"  {wmean:13.2f}  {ts:>12}"
    print(row)
