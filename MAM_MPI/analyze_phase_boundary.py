#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析相变边界随 gL 的演化，检验 gL 增大是'软化'还是'平移'相变。"""
import json
import numpy as np

d = json.load(open('/tmp/vipgl_2d_grid_result.json'))
gl = [float(x) for x in d['gl']]
vipext = [float(x) for x in d['vipext']]
V4 = np.array(d['rates']['V4'])
S4 = np.array(d['rates']['S4'])

print("=== S4 在 vipext=0.25/0.30/0.35/0.40 随 gL 变化 ===")
for v in [0.25, 0.30, 0.35, 0.40]:
    col = [S4[i, vipext.index(v)] for i in range(len(gl))]
    print(f"vipext={v:.2f}: " + " ".join(f"{x:>5.1f}" for x in col))

print()
print("=== 相变幅度对比 (V4 低态->高态跳变倍数) ===")
for g_i in [0, 5, 10, 15, 20]:
    g = gl[g_i]
    lo = V4[g_i, vipext.index(0.25)]
    hi = V4[g_i, vipext.index(0.35)]
    print(f"gL={g}: V4@0.25={lo:.1f}Hz -> V4@0.35={hi:.1f}Hz (跳变 {hi/max(lo,0.01):.0f}倍)")

print()
print("=== S4 相变幅度对比 ===")
for g_i in [0, 5, 10, 15, 20]:
    g = gl[g_i]
    hi = S4[g_i, vipext.index(0.25)]
    lo = S4[g_i, vipext.index(0.35)]
    print(f"gL={g}: S4@0.25={hi:.1f}Hz -> S4@0.35={lo:.1f}Hz (下降 {hi/max(lo,0.01):.0f}倍)")

print()
print("=== 相变边界位置(首次>10Hz 或 <10Hz 的 vipext) ===")
for i, g in enumerate(gl):
    row = V4[i]
    idx = np.where(row > 10)[0]
    thresh = f"vipext={vipext[idx[0]]:.2f}" if len(idx) > 0 else "无(全程低态)"
    print(f"gL={g:>4}: V4 相变边界在 {thresh}")
