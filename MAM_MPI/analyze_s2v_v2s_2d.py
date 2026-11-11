#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第四层 SST<->VIP 互抑权重二维扫描分析。
sst2vip (S4->V4) ∈ [0,1]，vip2sst (V4->S4) ∈ [0,1]，步长 0.05，21×21=441 组合。
提取各群体放电率，重点看 S4/V4/E4 的 winner-take-all 相变结构。
"""
import os, json
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']

S2V_LIST = [round(0.05 * i, 2) for i in range(21)]   # 0.00~1.00
V2S_LIST = [round(0.05 * i, 2) for i in range(21)]   # 0.00~1.00

def rate(sdir, p):
    f = os.path.join(sdir, "V1", f"V1_{p}_spikes.csv")
    if not os.path.exists(f):
        return 0.0
    a = np.loadtxt(f, delimiter=',')
    if a.size == 0:
        return 0.0
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a.shape[0] / nn[p] / DUR_S

def fmt(x):
    return str(x)   # 与 model_name 的 str(args.xxx) 一致

def find_dir(s2v, v2s):
    name = "spike_0.5s_AreaNum1_ext0.02"
    if s2v != 1.0:
        name += f"_s2v{fmt(s2v)}"
    if v2s != 1.0:
        name += f"_v2s{fmt(v2s)}"
    return os.path.join(BASE, name + "_GPU1")

# 校验缺失
missing = []
for s2v in S2V_LIST:
    for v2s in V2S_LIST:
        if not os.path.exists(find_dir(s2v, v2s)):
            missing.append((s2v, v2s))
print(f"总组合: {len(S2V_LIST)*len(V2S_LIST)}, 缺失: {len(missing)}")
for m in missing[:30]:
    print("  missing:", m)

# 每个群体一个 (s2v=21, v2s=21) 矩阵
out = {}
for p in pops:
    mat = np.array([[rate(find_dir(s2v, v2s), p) for v2s in V2S_LIST] for s2v in S2V_LIST])
    out[p] = mat

np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/s2v_v2s_2d_matrices.npz", **out)
np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/s2v_v2s_2d_meta.npz",
         s2v_list=np.array(S2V_LIST), v2s_list=np.array(V2S_LIST), pops=np.array(pops))
print("saved s2v_v2s_2d_matrices.npz")
print(f"sst2vip: {S2V_LIST[0]}~{S2V_LIST[-1]} ({len(S2V_LIST)})")
print(f"vip2sst: {V2S_LIST[0]}~{V2S_LIST[-1]} ({len(V2S_LIST)})")

# 关键群体摘要：S4(V4) 在 (s2v=0,v2s=0) 完全关断互抑时的放电率
for p in ['S4', 'V4', 'E4', 'P4']:
    m = out[p]
    print(f"{p}: 全关断(0,0)={m[0,0]:.2f} Hz, 基线(1,1)={m[-1,-1]:.2f} Hz, "
          f"只关S2V(0,1)={m[0,-1]:.2f} Hz, 只关V2S(1,0)={m[-1,0]:.2f} Hz")
