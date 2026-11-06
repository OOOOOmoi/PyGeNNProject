#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 VIP gL × vipext 二维网格扫描：提取全部 17 群体放电率矩阵，输出 JSON 供热力图。"""
import os, json
import numpy as np

SPIKE_ROOT = "output/spike"
DATA_JSON = "../model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

POPS = ['H1','E23','P23','S23','V23','E4','P4','S4','V4','E5','P5','S5','V5','E6','P6','S6','V6']

# 与 getModelName 的 float 格式化一致
def fmt(x):
    return str(float(x))

def find_dir(vipext, gl):
    tag = "0.5s_AreaNum1_ext0.02"
    if float(vipext) != 1.0:
        tag += f"_vipext{fmt(vipext)}"
    if float(gl) != 1.0:
        tag += f"_vipgl{fmt(gl)}"
    tag += "_GPU1"
    return os.path.join(SPIKE_ROOT, f"spike_{tag}", "V1")

def pop_rate(spike_dir, pop, nn):
    f = os.path.join(spike_dir, f"V1_{pop}_spikes.csv")
    if not os.path.exists(f):
        return 0.0
    a = np.loadtxt(f, delimiter=',')
    if a.size == 0:
        return 0.0
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a.shape[0] / nn / DUR_S

d = json.load(open(DATA_JSON))
nn = d['neuron_numbers']['V1']

GL = [round(1.0 + 0.05*i, 2) for i in range(21)]     # 1.00~2.00
VIP = [round(0.05 + 0.05*i, 2) for i in range(20)]   # 0.05~1.00

# 收集所有组合的 17 群体放电率
result = {p: np.full((len(GL), len(VIP)), np.nan) for p in POPS}
missing = []
for i, gl in enumerate(GL):
    for j, v in enumerate(VIP):
        sd = find_dir(v, gl)
        if not os.path.isdir(sd):
            missing.append((gl, v))
            continue
        for p in POPS:
            result[p][i, j] = pop_rate(sd, p, nn[p])

# 输出统计
print(f"共 {len(GL)*len(VIP)} 组合，缺失 {len(missing)} 个")
if missing:
    print("缺失组合(前20):", missing[:20])

# 保存 JSON（用 list 便于前端读取）
out = {
    "gl": [str(g) for g in GL],
    "vipext": [str(v) for v in VIP],
    "pops": POPS,
    "rates": {p: result[p].tolist() for p in POPS},
}
json.dump(out, open("/tmp/vipgl_2d_grid_result.json", "w"))
print("结果已存 /tmp/vipgl_2d_grid_result.json")

# 打印关键群体 V4/S4 的矩阵摘要
for p in ['V4', 'S4']:
    print(f"\n=== {p} 放电率矩阵 (行=gL, 列=vipext) ===")
    print("      " + " ".join(f"{v:>5}" for v in VIP))
    for i, gl in enumerate(GL):
        print(f"{gl:>5} " + " ".join(f"{result[p][i,j]:>5.1f}" for j in range(len(VIP))))
