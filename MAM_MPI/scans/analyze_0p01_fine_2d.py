#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0.01 步长细扫：gL ∈ {1.0, 2.0}，vipext ∈ [0.20,0.50]，sstext ∈ [0.05,1.00]。"""
import os, json
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']

GL_LIST = [1.0, 2.0]
VIP_LIST = [round(0.20 + 0.01*i, 2) for i in range(31)]   # 0.20~0.50
SST_LIST = [round(0.05 + 0.01*i, 2) for i in range(96)]   # 0.05~1.00

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
    # 与 model_name 的 str(args.xxx) 一致：直接用 float 的默认 str 表示。
    # str(0.30)="0.3", str(0.31)="0.31", str(2.0)="2.0" —— 注意不能用 :g（会把 2.0 变成 "2"）。
    return str(x)

def find_dir(gl, v, s):
    name = "spike_0.5s_AreaNum1_ext0.02"
    if v != 1.0:
        name += f"_vipext{fmt(v)}"
    if s != 1.0:
        name += f"_sstext{fmt(s)}"
    if gl != 1.0:
        name += f"_vipgl{fmt(gl)}"
    return os.path.join(BASE, name + "_GPU1")

# 校验
missing = []
for gl in GL_LIST:
    for v in VIP_LIST:
        for s in SST_LIST:
            if not os.path.exists(find_dir(gl, v, s)):
                missing.append((gl, v, s))
print(f"总组合: {len(GL_LIST)*len(VIP_LIST)*len(SST_LIST)}, 缺失: {len(missing)}")
if missing:
    for m in missing[:30]:
        print("  missing:", m)

# 保存矩阵: 每个 (gl, pop) 一个 (vipext=31, sstext=96) 矩阵
out = {}
for gl in GL_LIST:
    for p in pops:
        mat = np.array([[rate(find_dir(gl, v, s), p) for s in SST_LIST] for v in VIP_LIST])
        out[f"gl{fmt(gl)}_{p}"] = mat

np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/fine_0p01_matrices.npz", **out)
np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/fine_0p01_meta.npz",
         vip_list=np.array(VIP_LIST), sst_list=np.array(SST_LIST),
         gl_list=np.array(GL_LIST), pops=np.array(pops))
print("saved fine_0p01_matrices.npz")
print(f"vipext: {VIP_LIST[0]}~{VIP_LIST[-1]} ({len(VIP_LIST)})")
print(f"sstext: {SST_LIST[0]}~{SST_LIST[-1]} ({len(SST_LIST)})")
