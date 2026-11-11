#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0.01 步长细扫：精确定位 gL=1.0 vs 2.0 的 V4/S4 相变边界。"""
import numpy as np

NPZ = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/fine_0p01_matrices.npz"
META = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/fine_0p01_meta.npz"

d = np.load(NPZ)
meta = np.load(META)
pops = [str(p) for p in meta['pops']]
VIP_LIST = [float(x) for x in meta['vip_list']]
SST_LIST = [float(x) for x in meta['sst_list']]
GL_LIST = [float(x) for x in meta['gl_list']]

def fmt(x):
    return str(x)  # str(2.0)="2.0"，不能用 :g（会把 2.0 变成 "2"）

def key(gl, p):
    return f"gl{fmt(gl)}_{p}"

print("=" * 80)
print("V4 相变边界（首次 >10Hz 的 vipext）随 sstext 变化，0.01 精度")
print("=" * 80)
print(f"{'sstext':>8} {'gL=1.0 边界':>14} {'gL=2.0 边界':>14} {'差值':>8}")

for j, s in enumerate(SST_LIST):
    b1 = b2 = np.nan
    for gl in [1.0, 2.0]:
        col = d[key(gl, 'V4')][:, j]  # (vipext,) 列
        idx = np.where(col > 10)[0]
        b = VIP_LIST[idx[0]] if len(idx) > 0 else np.nan
        if gl == 1.0:
            b1 = b
        else:
            b2 = b
    # 只在 sstext 为 0.05 步长整数点打印（每 5 个打印一次，避免 96 行）
    if j % 5 == 0 or s in [0.05, 0.30, 0.50, 1.00]:
        diff = (b2 - b1) if not (np.isnan(b1) or np.isnan(b2)) else np.nan
        print(f"{s:>8.2f} {b1 if not np.isnan(b1) else 999:>14.2f} {b2 if not np.isnan(b2) else 999:>14.2f} {diff if not np.isnan(diff) else 0:>8.2f}")

print()
print("=" * 80)
print("S4 相变边界（首次 <20Hz 的 vipext，即从高态跌落）随 sstext 变化")
print("=" * 80)
print(f"{'sstext':>8} {'gL=1.0 边界':>14} {'gL=2.0 边界':>14} {'差值':>8}")

for j, s in enumerate(SST_LIST):
    b1 = b2 = np.nan
    for gl in [1.0, 2.0]:
        col = d[key(gl, 'S4')][:, j]
        idx = np.where(col < 20)[0]
        b = VIP_LIST[idx[0]] if len(idx) > 0 else np.nan
        if gl == 1.0:
            b1 = b
        else:
            b2 = b
    if j % 5 == 0 or s in [0.05, 0.30, 0.50, 1.00]:
        diff = (b2 - b1) if not (np.isnan(b1) or np.isnan(b2)) else np.nan
        print(f"{s:>8.2f} {b1 if not np.isnan(b1) else 999:>14.2f} {b2 if not np.isnan(b2) else 999:>14.2f} {diff if not np.isnan(diff) else 0:>8.2f}")

print()
# 汇总：gL 对相变边界的整体平移量
print("=" * 80)
print("gL 对 V4 相变边界的整体平移（平均差值，负值=边界左移/下移）")
print("=" * 80)
diffs = []
for j in range(len(SST_LIST)):
    b1 = b2 = np.nan
    col1 = d[key(1.0, 'V4')][:, j]
    col2 = d[key(2.0, 'V4')][:, j]
    i1 = np.where(col1 > 10)[0]
    i2 = np.where(col2 > 10)[0]
    if len(i1) > 0 and len(i2) > 0:
        diffs.append(VIP_LIST[i2[0]] - VIP_LIST[i1[0]])
if diffs:
    print(f"平均平移: {np.mean(diffs):+.4f} (n={len(diffs)})")
    print(f"标准差:   {np.std(diffs):.4f}")
    print(f"范围:     [{min(diffs):+.2f}, {max(diffs):+.2f}]")
