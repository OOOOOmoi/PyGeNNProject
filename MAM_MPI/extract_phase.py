#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取关键群体相变边界：找出每个 vipext 行里 S4 从低到高的突变 sstext。"""
import numpy as np

d = np.load("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/hires_matrices.npz")
VALS = [round(0.05*i, 2) for i in range(1, 21)]

S4 = d['S4']; V4 = d['V4']; P4 = d['P4']; V5 = d['V5']; V6 = d['V6']

print("=== S4 相变边界检测 (阈值 30 Hz) ===")
print(f"{'vipext':>8} {'sstext边界':>10}  {'S4低于/高于边界':>20}")
for i, v in enumerate(VALS):
    row = S4[i]  # 随 sstext 变化
    # 找第一个 S4>30 的 sstext 索引
    idx = None
    for j, s in enumerate(VALS):
        if row[j] > 30:
            idx = j
            break
    if idx is None:
        print(f"{v:>8}  无相变(全低)   max={row.max():.1f}")
    else:
        s_bound = VALS[idx]
        print(f"{v:>8}  {s_bound:>10}  [{row[0]:.1f} -> {row[idx]:.1f}]")

print("\n=== 关键群体在几个代表点的放电率 ===")
# 打印 (vipext, sstext) 对角线上几个点的 V4/S4/V5/V6
print(f"{'vipext=sstext':>12} {'V4':>8} {'S4':>8} {'P4':>8} {'V5':>8} {'V6':>8}")
for i in range(20):
    v = VALS[i]
    print(f"{v:>12} {V4[i,i]:8.2f} {S4[i,i]:8.2f} {P4[i,i]:8.2f} {V5[i,i]:8.2f} {V6[i,i]:8.2f}")
