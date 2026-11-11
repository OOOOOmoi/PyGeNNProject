#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四层 SST<->VIP 互抑权重二维扫描热图。
x = vip2sst (V4->S4)，y = sst2vip (S4->V4)，各 0~1 步长 0.05。
重点展示 S4/V4 的 winner-take-all 结构，E4/P4 作参考。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPZ = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/s2v_v2s_2d_matrices.npz"
META = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/s2v_v2s_2d_meta.npz"
OUT = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/s2v_v2s_2d_heatmap.png"

d = np.load(NPZ)
meta = np.load(META)
pops = [str(p) for p in meta['pops']]
S2V = [float(x) for x in meta['s2v_list']]
V2S = [float(x) for x in meta['v2s_list']]

type_color = {"H":"#8B5CF6","E":"#EF4444","S":"#3B82F6","P":"#22C55E","V":"#F59E0B"}

# 主图：S4 / V4 / E4 / P4 四个关键群体 2×2
key_pops = ['V4', 'S4', 'E4', 'P4']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, p in zip(axes.ravel(), key_pops):
    mat = d[p]  # (s2v, v2s) 矩阵，行=s2v(0~1)，列=v2s(0~1)
    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=0.05, vmax=100), interpolation='bilinear',
                   extent=[V2S[0], V2S[-1], S2V[0], S2V[-1]])
    ax.set_xlabel("vip2sst weight (V4->S4)", fontsize=11)
    ax.set_ylabel("sst2vip weight (S4->V4)", fontsize=11)
    ax.set_title(f"{p}", fontsize=14, fontweight='bold', color=type_color.get(p[0], 'black'), pad=6)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("firing rate (Hz)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
fig.suptitle("Layer-4 SST<->VIP mutual inhibition 2D scan (firing rate Hz, log scale)",
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")

# 完整 17 群体分面图
fig2, axes2 = plt.subplots(3, 6, figsize=(26, 13))
axes2 = axes2.ravel()
for i, p in enumerate(pops):
    ax = axes2[i]
    mat = d[p]
    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=0.05, vmax=100), interpolation='bilinear',
                   extent=[V2S[0], V2S[-1], S2V[0], S2V[-1]])
    ax.set_xlabel("vip2sst", fontsize=8)
    ax.set_ylabel("sst2vip", fontsize=8)
    ax.set_title(p, fontsize=12, fontweight='bold', color=type_color.get(p[0], 'black'), pad=4)
    ax.tick_params(labelsize=6)
ax = axes2[17]
ax.axis('off')
ax.text(0.5, 0.70, "Layer-4 SST<->VIP", ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.62, "mutual inhibition", ha='center', fontsize=13, transform=ax.transAxes)
ax.text(0.5, 0.52, "21 s2v x 21 v2s", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.44, "both 0.00~1.00", ha='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.36, "inferno (log), Hz", ha='center', fontsize=9, transform=ax.transAxes)
for j, (lab, c) in enumerate([("H","#8B5CF6"),("E","#EF4444"),("S","#3B82F6"),("P","#22C55E"),("V","#F59E0B")]):
    ax.text(0.5, 0.26 - j*0.05, lab, ha='center', fontsize=9, color=c, fontweight='bold', transform=ax.transAxes)
fig2.suptitle("V1 firing rate (Hz): layer-4 SST->VIP x VIP->SST weight scan (all 17 pops)",
              fontsize=16, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig(OUT.replace(".png", "_all17.png"), dpi=120)
print(f"saved {OUT.replace('.png', '_all17.png')}")
