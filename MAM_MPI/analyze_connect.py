#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 V1 各群体接收的兴奋/抑制突触数、连接度，诊断 S/P 群体放电率异常原因。"""
import json

data = json.load(open('../model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json'))
sn = data['synapses']['V1']
nn = data['neuron_numbers']['V1']
pops = data['population_list']

print("=== 各群体接收的突触数（每神经元平均，仅 V1 内部）===")
print(f"{'tar':5s} {'E-syn/neu':>10s} {'I-syn/neu':>10s} {'ext-syn/neu':>11s} {'E占比':>6s}")
for p in pops:
    e_syn = sum(sn[p]['V1'][s] for s in pops if s[0] == 'E')
    i_syn = sum(sn[p]['V1'][s] for s in pops if s[0] in 'SPVH')
    ext = sn[p]['external']['external']
    n = nn[p]
    total = e_syn + i_syn
    print(f"{p:5s} {e_syn/n:10.1f} {i_syn/n:10.1f} {ext/n:11.1f} {e_syn/total*100:5.1f}%")

print()
print("=== S 群体的兴奋性突触来源分布（谁驱动 SST？）===")
for p in ['S23', 'S4', 'S5', 'S6']:
    row = []
    for s in pops:
        if s[0] == 'E' and sn[p]['V1'][s] > 0:
            row.append(f"{s}:{sn[p]['V1'][s]/nn[p]:.0f}")
    print(f"  {p} 兴奋输入 <- {', '.join(row)}")

print()
print("=== S4 vs E4 接收的抑制性突触来源（每神经元）===")
for p in ['S4', 'E4']:
    row = []
    for s in pops:
        if s[0] in 'SPVH' and sn[p]['V1'][s] > 0:
            row.append(f"{s}:{sn[p]['V1'][s]/nn[p]:.1f}")
    print(f"  {p} 抑制输入 <- {', '.join(row)}")
