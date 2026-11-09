#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0.01 步长细扫热图：对比 gL=1.0 vs gL=2.0，展示 17 群体（V4/S4 相变区高亮）。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPZ = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/fine_0p01_matrices.npz"
META = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/fine_0p01_meta.npz"
OUT = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/grid_0p01_fine_heatmap.png"

d = np.load(NPZ)
meta = np.load(META)
pops = [str(p) for p in meta['pops']]
VIP_LIST = [float(x) for x in meta['vip_list']]
SST_LIST = [float(x) for x in meta['sst_list']]
GL_LIST = [float(x) for x in meta['gl_list']]

def fmt(x):
    return str(x)  # str(2.0)="2.0"，不能用 :g（会把 2.0 变成 "2"）

type_color = {"H":"#8B5CF6","E":"#EF4444","S":"#3B82F6","P":"#22C55E","V":"#F59E0B"}

# 两张大图：gL=1.0 和 gL=2.0，各 17 群体分面
for gl in GL_LIST:
    fig, axes = plt.subplots(3, 6, figsize=(26, 13))
    axes = axes.ravel()
    key = lambda p: f"gl{fmt(gl)}_{p}"
    for i, p in enumerate(pops):
        ax = axes[i]
        mat = d[key(p)]
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap='inferno',
                       norm=LogNorm(vmin=0.05, vmax=100), interpolation='bilinear',
                       extent=[SST_LIST[0], SST_LIST[-1], VIP_LIST[0], VIP_LIST[-1]])
        ax.set_xlabel("sstext", fontsize=8)
        ax.set_ylabel("vipext", fontsize=8)
        ax.set_title(p, fontsize=12, fontweight='bold', color=type_color.get(p[0],'black'), pad=4)
        ax.tick_params(labelsize=6)
    ax = axes[17]
    ax.axis('off')
    ax.text(0.5, 0.68, "0.01 step grid", ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.58, f"gL = {fmt(gl)}", ha='center', fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.50, "31 vipext x 96 sstext", ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.42, "vipext 0.20~0.50, sstext 0.05~1.00", ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.34, "inferno (log), Hz", ha='center', fontsize=9, transform=ax.transAxes)
    for j, (lab, c) in enumerate([("H1","#8B5CF6"),("E","#EF4444"),("S","#3B82F6"),("P","#22C55E"),("V","#F59E0B")]):
        ax.text(0.5, 0.24 - j*0.06, lab, ha='center', fontsize=9, color=c, fontweight='bold', transform=ax.transAxes)
    fig.suptitle(f"V1 firing rate (Hz): 0.01-step VIP x SST grid, gL={fmt(gl)}", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT.replace(".png", f"_gl{fmt(gl)}.png"), dpi=120)
    print(f"saved grid_0p01_fine_heatmap_gl{fmt(gl)}.png")

# 额外：V4 和 S4 在两种 gL 下的并排对比图
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
for row, p in enumerate(['V4', 'S4']):
    for col, gl in enumerate(GL_LIST):
        ax = axes2[row][col]
        mat = d[key(p)]
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap='inferno',
                       norm=LogNorm(vmin=0.05, vmax=100), interpolation='bilinear',
                       extent=[SST_LIST[0], SST_LIST[-1], VIP_LIST[0], VIP_LIST[-1]])
        ax.set_xlabel("sstext", fontsize=10)
        ax.set_ylabel("vipext", fontsize=10)
        ax.set_title(f"{p}  gL={fmt(gl)}", fontsize=12, fontweight='bold',
                     color=type_color.get(p[0],'black'))
        ax.tick_params(labelsize=8)
fig2.suptitle("V4 & S4 firing rate (Hz): gL=1.0 vs gL=2.0 (0.01 step)", fontsize=15, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/grid_0p01_V4S4_compare.png", dpi=120)
print("saved grid_0p01_V4S4_compare.png")
