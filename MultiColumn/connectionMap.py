import numpy as np
import matplotlib.pyplot as plt
import os
import json
from itertools import product

# 路径加载
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
DataPath = os.path.join(parent_dir, "custom_Data_Model_3396.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)

AreaList = ["V1"]
PopList = ParamOfAll["population_list"]
SynapsesWeightMean = ParamOfAll["synapse_weights_mean"]
SynapsesNumber = ParamOfAll["synapses"]

# 构造 nodes 列表，保持人口顺序
nodes = [f"{area}_{pop}" for area in AreaList for pop in PopList]
node_idx = {node: i for i, node in enumerate(nodes)}
n_nodes = len(nodes)

# 初始化连接矩阵，默认填充 NaN（代表无连接或 0）
matrix = np.full((n_nodes, n_nodes), np.nan)

# 填充矩阵
for srcArea, tarArea in product(AreaList, AreaList):
    for srcPop, tarPop in product(PopList, PopList):
        try:
            weight = SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] / 1000.0
            synNum = SynapsesNumber[tarArea][tarPop][srcArea][srcPop]
        except KeyError:
            continue  # 跳过缺失的数据

        target_node = f"{tarArea}_{tarPop}"
        source_node = f"{srcArea}_{srcPop}"
        i = node_idx[target_node]
        j = node_idx[source_node]

        if synNum > 0:
            matrix[i, j] = synNum * weight  # 仅保留非零连接；否则仍为 NaN

# 归一化非零权重到 [0, 1]
non_zero_weights = matrix[~np.isnan(matrix)]
if len(non_zero_weights) > 0:
    min_weight = np.min(non_zero_weights)
    max_weight = np.max(non_zero_weights)
    normalized_matrix = (matrix - min_weight) / (max_weight - min_weight)
else:
    normalized_matrix = matrix  # 全 NaN

# 绘制热图
plt.figure(figsize=(12, 10))
masked_matrix = np.ma.masked_invalid(normalized_matrix)

cmap = plt.cm.viridis.copy()
cmap.set_bad('white', 1.0)  # NaN 显示为白色

plt.imshow(masked_matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
cbar = plt.colorbar(label="Normalized Weight")
cbar.set_ticks([0, 0.5, 1])

# 坐标轴设置
plt.xticks(np.arange(n_nodes), nodes, rotation=90)
plt.yticks(np.arange(n_nodes), nodes)
plt.xlabel("Source Node")
plt.ylabel("Target Node")
plt.title("Normalized Connection Matrix (Zero Synapses in White)")

plt.tight_layout()
plt.savefig("connection_map.jpg", dpi=300)