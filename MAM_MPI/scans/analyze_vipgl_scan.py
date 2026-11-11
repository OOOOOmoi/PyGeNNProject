#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 VIP gL 扫描实验结果：提取各组合 V4/S4 放电率，看相变是否随 vip-gl-scale 变缓。"""
import os, json
import numpy as np

SPIKE_ROOT = "output/spike"
DATA_JSON = "../model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

def fmt_vipext(v):
    # 与 getModelName 的 float 格式化一致
    return str(float(v))

def find_dir(vipext, gl):
    tag = f"0.5s_AreaNum1_ext0.02_vipext{fmt_vipext(vipext)}"
    if gl != 1.0:
        tag += f"_vipgl{gl}"
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

VIPEXT = [0.25, 0.30, 0.325, 0.35, 0.40]
GL = [1.0, 2.0, 4.0, 8.0]

print("=== V4 放电率 (Hz) 矩阵：行=vip-gl-scale, 列=vipext ===")
print(f"{'gl':>5} " + " ".join(f"{v:>7}" for v in VIPEXT))
v4_matrix = {}
for gl in GL:
    row = []
    for v in VIPEXT:
        sd = find_dir(v, gl)
        r = pop_rate(sd, 'V4', nn['V4'])
        row.append(r)
        v4_matrix[(gl, v)] = r
    print(f"{gl:>5} " + " ".join(f"{r:>7.2f}" for r in row))

print()
print("=== S4 放电率 (Hz) 矩阵 ===")
print(f"{'gl':>5} " + " ".join(f"{v:>7}" for v in VIPEXT))
for gl in GL:
    row = [pop_rate(find_dir(v, gl), 'S4', nn['S4']) for v in VIPEXT]
    print(f"{gl:>5} " + " ".join(f"{r:>7.2f}" for r in row))

# 输出 JSON 供绘图
out = {
    "vipext": VIPEXT,
    "gl": GL,
    "V4": {f"{gl}": [v4_matrix[(gl, v)] for v in VIPEXT] for gl in GL},
    "S4": {f"{gl}": [pop_rate(find_dir(v, gl), 'S4', nn['S4']) for v in VIPEXT] for gl in GL},
}
json.dump(out, open("/tmp/vipgl_scan_result.json", "w"))
print("\n结果已存 /tmp/vipgl_scan_result.json")
