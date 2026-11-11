#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gL=2.0 下的 vip-ext × sst-ext 二维热图（20×20）：展示全部 17 群体。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPZ = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/vipgl2_hires_matrices.npz"
OUT = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/grid_vipgl2_heatmap.png"

d = np.load(NPZ)
pops = [k for k in d.files]
VALS = [round(0.05*i, 2) for i in range(1, 21)]
tick_labels = [f"{v:g}" for v in VALS]

type_color = {"H":"#8B5CF6","E":"#EF4444","S":"#3B82F6","P":"#22C55E","V":"#F59E0B"}

fig, axes = plt.subplots(3, 6, figsize=(24, 12))
axes = axes.ravel()

for i, p in enumerate(pops):
    ax = axes[i]
    mat = d[p]
    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=0.05, vmax=100), interpolation='bilinear')
    step = 2
    ax.set_xticks(range(0, 20, step))
    ax.set_xticklabels([tick_labels[j] for j in range(0, 20, step)], fontsize=6)
    ax.set_yticks(range(0, 20, step))
    ax.set_yticklabels([tick_labels[j] for j in range(0, 20, step)], fontsize=6)
    ax.set_xlabel("sstext", fontsize=8)
    ax.set_ylabel("vipext", fontsize=8)
    ax.set_title(p, fontsize=12, fontweight='bold', color=type_color.get(p[0],'black'), pad=4)

ax = axes[17]
ax.axis('off')
ax.text(0.5, 0.68, "VIP x SST grid", ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.58, "gL = 2.0", ha='center', fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.50, "400 combos (0.05 step)", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.42, "ext=0.02, 500ms", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.34, "inferno (log), Hz", ha='center', fontsize=9, transform=ax.transAxes)
for j, (lab, c) in enumerate([("H1","#8B5CF6"),("E","#EF4444"),("S","#3B82F6"),("P","#22C55E"),("V","#F59E0B")]):
    ax.text(0.5, 0.24 - j*0.06, lab, ha='center', fontsize=9, color=c, fontweight='bold', transform=ax.transAxes)

fig.suptitle("V1 firing rate (Hz): VIP x SST grid with gL=2.0 (0.05 step, 400 combos)", fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=120)
print("saved", OUT)
