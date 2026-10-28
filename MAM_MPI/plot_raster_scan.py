#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为每个参数组合的 spike CSV 生成 V1 raster plot（按群体分层着色）。"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
OUT  = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/raster_scan"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_MS = 500.0

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']

# 群体颜色：按类型 H/E/S/P/V 上色
type_color = {"H": "#8B5CF6", "E": "#EF4444", "S": "#3B82F6", "P": "#22C55E", "V": "#F59E0B"}

def pop_color(p):
    return type_color.get(p[0], "gray")

def plot_one(sdir, tag, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    y0 = 0
    ticks, labels = [], []
    for p in pops:
        f = os.path.join(sdir, "V1", f"V1_{p}_spikes.csv")
        N = nn[p]
        y_positions = []
        times = []
        if os.path.exists(f):
            try:
                a = np.loadtxt(f, delimiter=',')
            except Exception:
                a = np.empty((0, 2))
            if a.size == 0:
                a = np.empty((0, 2))
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape[0] > 0:
                times = a[:, 0]
                ids = a[:, 1].astype(int)
                # 每个神经元在本群体内的相对 y 偏移
                y_positions = y0 + ids / max(N - 1, 1) * 0.9
        else:
            times = []
        if len(times) > 0:
            ax.scatter(times, y_positions, s=0.5, c=pop_color(p), lw=0, rasterized=True)
        ticks.append(y0 + 0.45)
        labels.append(p)
        y0 += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, DUR_MS)
    ax.set_ylim(-0.2, len(pops) + 0.2)
    ax.set_xlabel("Time (ms)", fontsize=13)
    ax.set_ylabel("Population", fontsize=13)
    ax.set_title(f"V1 Raster - {tag}", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved {out_path}")

os.makedirs(OUT, exist_ok=True)

# 扫描所有 spike_0.5s_AreaNum1_* 目录
dirs = sorted(glob.glob(os.path.join(BASE, "spike_0.5s_AreaNum1_*_GPU1")))
if not dirs:
    print("no dirs found")
for sdir in dirs:
    name = os.path.basename(sdir).replace("spike_0.5s_AreaNum1_", "").replace("_GPU1", "")
    out_path = os.path.join(OUT, f"raster_V1_{name}.png")
    print(f"[{name}]")
    plot_one(sdir, name, out_path)
print("done, total", len(dirs))
