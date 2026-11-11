#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全局外部输入 ext-scale 精细扫描分析。
ext-scale ∈ [0, 0.1] 步长 0.001，共 101 值，其它参数默认 1.0。
提取 17 群体放电率随 ext-scale 的变化曲线，观察外部输入的群体响应。
"""
import os, json
import numpy as np

BASE = "/home/yangjinhao/PyGeNNProject/MAM_MPI/output/spike"
DATA = "/home/yangjinhao/PyGeNNProject/model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json"
DUR_S = 0.5

d = json.load(open(DATA))
nn = d['neuron_numbers']['V1']
pops = d['population_list']

EXT_LIST = [round(0.001 * i, 3) for i in range(101)]   # 0.0~0.1

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
    return str(x)   # 与 getModelName 的 str(args.ext_scale) 一致

def find_dir(ext):
    name = "spike_0.5s_AreaNum1"
    if ext != 1.0:
        name += f"_ext{fmt(ext)}"
    return os.path.join(BASE, name + "_GPU1")

# 校验缺失
missing = []
for ext in EXT_LIST:
    if not os.path.exists(find_dir(ext)):
        missing.append(ext)
print(f"总组合: {len(EXT_LIST)}, 缺失: {len(missing)}")
for m in missing[:30]:
    print("  missing:", m)

# 每个群体一条 (ext=101,) 曲线
out = {}
for p in pops:
    out[p] = np.array([rate(find_dir(ext), p) for ext in EXT_LIST])

np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/ext_fine_curves.npz", **out)
np.savez("/home/yangjinhao/PyGeNNProject/MAM_MPI/output/ext_fine_meta.npz",
         ext_list=np.array(EXT_LIST), pops=np.array(pops))
print("saved ext_fine_curves.npz")
print(f"ext-scale: {EXT_LIST[0]}~{EXT_LIST[-1]} ({len(EXT_LIST)})")

# 摘要：ext=0 和 ext=0.1 时各群体放电率
print("\next=0.0 时各群体放电率(Hz):")
for p in pops:
    print(f"  {p}: {out[p][0]:.3f}")
print("\next=0.1 时各群体放电率(Hz):")
for p in pops:
    print(f"  {p}: {out[p][-1]:.3f}")
