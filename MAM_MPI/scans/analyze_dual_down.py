#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同步双降 (vipext=sstext) 扫描：展示全部 17 群体放电率变化。"""
import os, json
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']   # 17 群体

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

# scale=1.0 是基线(无后缀), 其余双降
scales = ["1.0", "0.7", "0.5", "0.4", "0.3", "0.2", "0.1", "0.05"]

def find_dir(s):
    if s == "1.0":
        return os.path.join(BASE, "spike_0.5s_AreaNum1_ext0.02_GPU1")
    return os.path.join(BASE, f"spike_0.5s_AreaNum1_ext0.02_vipext{float(s):g}_sstext{float(s):g}_GPU1")

print("scale  " + "  ".join(f"{p:>6}" for p in pops))
for s in scales:
    sdir = find_dir(s)
    if not os.path.exists(sdir):
        print(f"{s:>5}  (missing: {os.path.basename(sdir)})")
        continue
    rates = [rate(sdir, p) for p in pops]
    print(f"{s:>5}  " + "  ".join(f"{r:6.2f}" for r in rates))
