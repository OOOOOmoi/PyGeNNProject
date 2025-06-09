import numpy as np
import matplotlib.pyplot as plt
import os
import json
from itertools import product
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)

AreaList=ParamOfAll['area_list']
PopList=ParamOfAll["population_list"]
SynapsesWeightMean=ParamOfAll["synapse_weights_mean"]
SynapsesNumber=ParamOfAll["synapses"]
# 读取数据，忽略零权重
filename = "SynapsesNumber.txt"
data = []
nodes = set()

for srcArea, tarArea in product(AreaList, AreaList):
    for srcPop, tarPop in product(PopList, PopList):
        weight = SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop]/1000.0
        synNum = SynapsesNumber[tarArea][tarPop][srcArea][srcPop]
        if synNum == 0.0:  # 忽略零权重
            continue
        
        target_node = f"{tarArea}_{tarPop}"
        source_node = f"{srcArea}_{srcPop}"
        nodes.add(target_node)
        nodes.add(source_node)
        data.append((target_node, source_node, synNum*weight))

# 排序节点列表
nodes = sorted(nodes)
node_idx = {node: i for i, node in enumerate(nodes)}
n_nodes = len(nodes)

# 初始化连接矩阵（默认填充 NaN，便于后续忽略零权重）
matrix = np.full((n_nodes, n_nodes), np.nan)

# 填充矩阵（仅非零权重）
for target, source, weight in data:
    i = node_idx[target]
    j = node_idx[source]
    matrix[i, j] = weight

# 归一化非零权重到 [0, 1]
non_zero_weights = matrix[~np.isnan(matrix)]
if len(non_zero_weights) > 0:
    min_weight = np.min(non_zero_weights)
    max_weight = np.max(non_zero_weights)
    # 归一化
    normalized_weights = (non_zero_weights - min_weight) / (max_weight - min_weight)

    # 将 0 替换为 NaN（隐藏）
    normalized_weights[normalized_weights == 0] = np.nan

    # 应用回矩阵
    matrix[~np.isnan(matrix)] = normalized_weights


plt.figure(figsize=(12, 10))

# 使用 masked array 隐藏 NaN（零权重）
masked_matrix = np.ma.masked_invalid(matrix)
cmap = plt.cm.viridis
cmap.set_bad('white', 1.0)  # 将 NaN 显示为白色

# 绘制热图
plt.imshow(masked_matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
cbar = plt.colorbar(label="Normalized Weight")
cbar.set_ticks([0, 0.5, 1])  # 标准化后的刻度

# 设置坐标轴
plt.xticks(np.arange(n_nodes), nodes, rotation=90)
plt.yticks(np.arange(n_nodes), nodes)
plt.xlabel("Source Node")
plt.ylabel("Target Node")
plt.title("Normalized Connection Matrix (Zero Weights Ignored)")

plt.tight_layout()
plt.savefig("connection_map.jpg", dpi=300)