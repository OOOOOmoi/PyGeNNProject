import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import itertools

def get_custom_colormap():
    base = plt.get_cmap("viridis")
    colors = base(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  # 将最小值设为白色
    return ListedColormap(colors)

def connectom(suffix, synapse_number, synapse_weight, neuron_number, arealist, layerlist, poplist, title='Synaptic Connectivity Overview'):
    pops = [f"{area}-{pop}-{layer}" for area, layer, pop in itertools.product(arealist, layerlist, poplist)]

    n_pops = len(pops)

    weight_matrix = np.zeros((n_pops, n_pops))
    indegree_matrix = np.zeros((n_pops, n_pops))
    effective_matrix = np.zeros((n_pops, n_pops))

    weight_mask = np.zeros((n_pops, n_pops), dtype=bool)
    indegree_mask = np.zeros((n_pops, n_pops), dtype=bool)
    effective_mask = np.zeros((n_pops, n_pops), dtype=bool)

    # ✅ 用于保存每行有效权重（非绝对值）求和
    effective_sum_vector = np.zeros(n_pops)

    for i, tarKey in enumerate(pops):
        tarArea, tarPop, tarLayer = tarKey.split('-')
        tar = (tarArea, tarLayer, tarPop)
        for j, srcKey in enumerate(pops):
            srcArea, srcPop, srcLayer = srcKey.split('-')
            src = (srcArea, srcLayer, srcPop)
            try:
                n_syn = synapse_number.loc[tar, src]
                w = synapse_weight.loc[tar, src]
                n_neuron = neuron_number.loc[tar]

                if n_syn == 0 or n_neuron == 0:
                    raise ValueError

                indegree = n_syn / n_neuron
                weight_matrix[i, j] = abs(w)
                indegree_matrix[i, j] = indegree
                effective = w * indegree / 1000.0  # ✅ 不取绝对值
                effective_matrix[i, j] = abs(effective)
                effective_sum_vector[i] += effective  # ✅ 保留符号求和

            except:
                weight_mask[i, j] = True
                indegree_mask[i, j] = True
                effective_mask[i, j] = True

    def normalize(mat, mask):
        m = np.copy(mat)
        if np.any(~mask):
            max_val = np.max(m[~mask])
            if max_val > 0:
                m = m / max_val
        return m

    weight_norm = normalize(weight_matrix, weight_mask)
    indegree_norm = normalize(indegree_matrix, indegree_mask)
    effective_norm = normalize(effective_matrix, effective_mask)

    fig, axes = plt.subplots(1, 3, figsize=(26, 7))  # ✅ 宽度略增
    titles = ['Effective Weight (W × In-degree)', 'Synaptic Weight (W)', 'In-degree (Connections/Neuron)']
    matrices = [effective_matrix, weight_matrix, indegree_matrix]
    masks = [effective_mask, weight_mask, indegree_mask]
    raw_values = [effective_matrix, weight_matrix, indegree_matrix]

    for ax, norm_mat, raw_mat, msk, ttl in zip(axes, matrices, raw_values, masks, titles):
        if ttl.startswith("Effective"):
            # ✅ 添加一列：每行的原始有效权重求和
            norm_mat = np.hstack([norm_mat, np.zeros((n_pops, 1))])  # dummy zeros (not plotted)
            raw_mat = np.hstack([raw_mat, effective_sum_vector.reshape(-1, 1)])
            msk = np.hstack([msk, np.zeros((n_pops, 1), dtype=bool)])
            xticklabels = pops + ['Σ']  # ✅ 添加最后一列标签
        else:
            xticklabels = pops

        sns.heatmap(
            norm_mat,
            mask=msk,
            annot=raw_mat.astype(int),
            fmt=".0f",
            xticklabels=xticklabels,
            yticklabels=pops,
            cmap=get_custom_colormap(),
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Normalized Value'},
            ax=ax
        )
        ax.set_title(ttl)
        ax.set_xlabel("Source Population")
        ax.set_ylabel("Target Population")
        ax.tick_params(axis='x', rotation=90)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"output/map/map_{suffix}.png")