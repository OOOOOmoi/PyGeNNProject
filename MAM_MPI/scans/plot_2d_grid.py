#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VIP × SST 二维网格热图：展示全部 17 群体放电率。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
OUT  = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/grid_2d_heatmap.png"
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
    name = "spike_0.5s_AreaNum1_ext0.02"
    if v != 1.0:
        name += f"_vipext{v:g}"
    if s != 1.0:
        name += f"_sstext{s:g}"
    return os.path.join(BASE, name + "_GPU1")

# 构建 17 群体 × 7×7 矩阵
M = {}
for p in pops:
    mat = np.array([[rate(find_dir(v, s), p) for s in SST] for v in VIP])
    M[p] = mat

# 分类型着色分组
type_color = {"H":"#8B5CF6","E":"#EF4444","S":"#3B82F6","P":"#22C55E","V":"#F59E0B"}

# 3 行 x 6 列 = 18 格，放 17 群体 + 1 个图例
fig, axes = plt.subplots(3, 6, figsize=(22, 11))
axes = axes.ravel()

for i, p in enumerate(pops):
    ax = axes[i]
    mat = M[p]
    im = ax.imshow(mat, aspect='auto', origin='lower',
                   cmap='inferno', norm=LogNorm(vmin=0.05, vmax=100))
    ax.set_xticks(range(len(SST)))
    ax.set_xticklabels([f"{s:g}" for s in SST], fontsize=7)
    ax.set_yticks(range(len(VIP)))
    ax.set_yticklabels([f"{v:g}" for v in VIP], fontsize=7)
    ax.set_xlabel("sstext", fontsize=8)
    ax.set_ylabel("vipext", fontsize=8)
    ax.set_title(f"{p}", fontsize=11, fontweight='bold',
                 color=type_color.get(p[0],'black'), pad=4)
    # 标注数值
    for r in range(len(VIP)):
        for c in range(len(SST)):
            ax.text(c, r, f"{mat[r,c]:.1f}", ha='center', va='center',
                    fontsize=5.5, color='white',
                    bbox=dict(boxstyle='round,pad=0.05', fc='black', alpha=0.25, lw=0))

# 最后一个格子放图例说明
ax = axes[17]
ax.axis('off')
ax.text(0.5, 0.65, "VIP x SST grid scan", ha='center', fontsize=13, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.55, "ext-scale = 0.02", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.45, "colormap = inferno (log)", ha='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.35, "color = firing rate (Hz)", ha='center', fontsize=9, transform=ax.transAxes)
# 图例
legend_items = [("H1", "#8B5CF6"), ("E (exc)", "#EF4444"), ("S (SST)", "#3B82F6"),
                ("P (PV)", "#22C55E"), ("V (VIP)", "#F59E0B")]
for j, (lab, c) in enumerate(legend_items):
    ax.text(0.5, 0.25 - j*0.06, lab, ha='center', fontsize=9, color=c, fontweight='bold', transform=ax.transAxes)

fig.suptitle("V1 firing rate (Hz): VIP x SST external input grid (ext=0.02, 500ms)", fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=110)
print("saved", OUT)
