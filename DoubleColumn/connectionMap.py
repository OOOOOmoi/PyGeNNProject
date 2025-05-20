import numpy as np
import matplotlib.pyplot as plt

# 读取数据，忽略零权重
filename = "SynapsesNumber.txt"
data = []
nodes = set()

with open(filename, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        target_region, target_pop, source_region, source_pop, weight = parts
        weight = float(weight)
        if weight == 0.0:  # 忽略零权重
            continue
        
        target_node = f"{target_region}_{target_pop}"
        source_node = f"{source_region}_{source_pop}"
        nodes.add(target_node)
        nodes.add(source_node)
        data.append((target_node, source_node, weight))

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
    matrix[~np.isnan(matrix)] = (non_zero_weights - min_weight) / (max_weight - min_weight)

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