#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全局外部输入 ext-scale 扫描绘图：17 群体放电率 vs ext-scale 曲线。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/ext_fine_curves.npz"
META = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/ext_fine_meta.npz"
OUT = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/ext_fine_curves.png"

d = np.load(NPZ)
meta = np.load(META)
pops = [str(p) for p in meta['pops']]
EXT = [float(x) for x in meta['ext_list']]

type_color = {"H":"#8B5CF6","E":"#EF4444","S":"#3B82F6","P":"#22C55E","V":"#F59E0B"}

# 主图：17 群体分面，每个群体一条放电率曲线
fig, axes = plt.subplots(3, 6, figsize=(26, 13))
axes = axes.ravel()
for i, p in enumerate(pops):
    ax = axes[i]
    ax.plot(EXT, d[p], color=type_color.get(p[0], 'black'), lw=1.8)
    ax.set_xlabel("ext-scale", fontsize=8)
    ax.set_ylabel("rate (Hz)", fontsize=8)
    ax.set_title(p, fontsize=12, fontweight='bold', color=type_color.get(p[0], 'black'), pad=4)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 0.1)
ax = axes[17]
ax.axis('off')
ax.text(0.5, 0.70, "ext-scale scan", ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.62, "0.000~0.100, step 0.001", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.52, "other params = 1.0", ha='center', fontsize=10, transform=ax.transAxes)
for j, (lab, c) in enumerate([("H","#8B5CF6"),("E","#EF4444"),("S","#3B82F6"),("P","#22C55E"),("V","#F59E0B")]):
    ax.text(0.5, 0.42 - j*0.05, lab, ha='center', fontsize=10, color=c, fontweight='bold', transform=ax.transAxes)
fig.suptitle("V1 firing rate vs global external input (ext-scale), all 17 pops", fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")

# 副图：按类型合并（E/S/P/V 各层平均），一张图看清类型差异
fig2, ax2 = plt.subplots(figsize=(11, 7))
for ctype in ['E', 'S', 'P', 'V', 'H']:
    pops_t = [p for p in pops if p[0] == ctype]
    curves = np.array([d[p] for p in pops_t])
    mean_c = curves.mean(axis=0)
    ax2.plot(EXT, mean_c, color=type_color[ctype], lw=2.2,
             label=f"{ctype} (avg over {len(pops_t)} pops)")
ax2.set_xlabel("ext-scale", fontsize=12)
ax2.set_ylabel("firing rate (Hz)", fontsize=12)
ax2.set_title("V1 firing rate vs ext-scale (type-averaged)", fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 0.1)
fig2.tight_layout()
fig2.savefig(OUT.replace(".png", "_bytype.png"), dpi=130)
print(f"saved {OUT.replace('.png', '_bytype.png')}")
