#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双降扫描：全部 17 群体放电率 vs scale 曲线图（英文标签）。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
OUT  = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/dual_down_rates.png"
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

def find_dir(s):
    if s == "1.0":
        return os.path.join(BASE, "spike_0.5s_AreaNum1_ext0.02_GPU1")
    return os.path.join(BASE, f"spike_0.5s_AreaNum1_ext0.02_vipext{float(s):g}_sstext{float(s):g}_GPU1")

scales = ["1.0","0.7","0.5","0.4","0.3","0.2","0.1","0.05"]
x = [float(s) for s in scales]

type_color = {"H":"#8B5CF6","E":"#EF4444","S":"#3B82F6","P":"#22C55E","V":"#F59E0B"}

fig, axes = plt.subplots(2, 1, figsize=(11, 12), sharex=True)

# 上：全部 17 群体
ax = axes[0]
for p in pops:
    y = [rate(find_dir(s), p) for s in scales]
    ax.plot(x, y, marker='o', ms=4, color=type_color.get(p[0],'gray'), label=p, lw=1.5)
ax.set_xscale('log')
ax.set_xlabel("vipext = sstext (log scale)", fontsize=13)
ax.set_ylabel("Firing rate (Hz)", fontsize=13)
ax.set_title("Dual-down (VIP+SST) scan: all 17 populations", fontsize=14, fontweight='bold')
ax.legend(ncol=4, fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# 下：重点群体（V4/V5/V6/S4/E4/P4）
ax2 = axes[1]
focus = ["V4","V5","V6","S4","S23","E4","P4","P23"]
for p in focus:
    y = [rate(find_dir(s), p) for s in scales]
    ax2.plot(x, y, marker='o', ms=5, color=type_color.get(p[0],'gray'), label=p, lw=2)
ax2.set_xscale('log')
ax2.set_xlabel("vipext = sstext (log scale)", fontsize=13)
ax2.set_ylabel("Firing rate (Hz)", fontsize=13)
ax2.set_title("Key populations: phase transition near scale=0.2", fontsize=14, fontweight='bold')
ax2.legend(ncol=4, fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", OUT)
