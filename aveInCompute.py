import numpy as np
from itertools import product
import os
import json
from collections import OrderedDict,defaultdict
Area=["V1"]
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir))
DataPath=os.path.join(parent_dir, "viscortex_raw_data.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)
Intrinsic_Connectivity=OrderedDict()
Intrinsic_Connectivity=ParamOfAll["Intrinsic_Connectivity"]

DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)
SynapsesWeightMean=OrderedDict()
PopList=OrderedDict()
NeuronNumber=OrderedDict()
SynapsesNumber=OrderedDict()
SynapsesWeightMean=ParamOfAll["synapse_weights_mean"]
PopList=ParamOfAll['population_list']
NeuronNumber=ParamOfAll["neuron_numbers"]
SynapsesNumber=ParamOfAll["synapses"]

def extract_subdict(param_dict, level1_key, level3_key):
    result = {}
    level1 = param_dict.get(level1_key, {})
    for level2_key, level2_val in level1.items():
        level3_dict = level2_val.get(level3_key)
        if level3_dict is not None:
            result[level2_key] = level3_dict
    return result

DataPath=os.path.join(parent_dir, "indegrees_full.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)
indegrees=OrderedDict()
indegrees=ParamOfAll
# indegrees = extract_subdict(ParamOfAll, "V1", "V1")

N=10.
F=(1-np.exp(-N))/N
R=40.#R=tau/C
taum=20.
C=0.5
EL=-70
Er=-60
dt=0.1
Fac = defaultdict(lambda: {"inSyn": 0.0, "offset": 0.0})
D=9.5
f=1
# 遍历所有的 target/source population 对组合
# 初始化矩阵
n = len(PopList)
ind_matrix = np.zeros((n, n))  # i: target, j: source

for i, tarpop in enumerate(PopList):
    for j, srcpop in enumerate(PopList):
        try:
            synNum = SynapsesNumber[Area[0]][tarpop][Area[0]][srcpop]
            neuronNum = NeuronNumber[Area[0]][tarpop]
            ind = synNum / neuronNum if neuronNum != 0 else 0
            ind_matrix[i, j] = ind
        except KeyError as e:
            print(f"Missing data for {tarpop}-{srcpop}: {e}")

C = dt * R * F
rhs_value = (9 * taum) / C - (EL - Er)  # 右边目标值 (对每个 i 相同)

# 构造 A 矩阵（系数矩阵）
A = np.copy(ind_matrix)
np.fill_diagonal(A, 0)  # 确保对角线是 0，即 j ≠ i 的限制

# 构造 b 向量（右侧结果）
b = np.full((17,), rhs_value)

# 求解 W
W = np.linalg.lstsq(A, b, rcond=None)[0]

for j, srcpop in enumerate(PopList):
    print(f"W_{srcpop} = {W[j]:.4f}")

for pop in PopList:
    W_ext = SynapsesWeightMean[Area[0]][pop]["external"]["external"]
    ind_ext = SynapsesNumber[Area[0]][pop]["external"]["external"] / NeuronNumber[Area[0]][pop]
    inSyn = W_ext * ind_ext * 0.5 * 10 / 1000.0
    Fac[pop]["offset"] = inSyn
# 保存 Fac 为 JSON 文件
output_path = os.path.join(parent_dir, "Fac_result.json")

# 将 defaultdict 转为普通字典，并确保所有值都是基本类型（float）
Fac_serializable = {k: {ik: float(iv) for ik, iv in v.items()} for k, v in Fac.items()}

with open(output_path, 'w') as f:
    json.dump(Fac_serializable, f, indent=4)

print(f"Fac 已保存到: {output_path}")