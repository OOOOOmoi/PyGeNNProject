import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def connectom(synapse_number, synapse_weight, neuron_number, structure, title='Synaptic Connectivity Overview'):
    pops = [f"{area}-{pop}" for area, pops in structure.items() for pop in pops]
    n_pops = len(pops)

    weight_matrix = np.zeros((n_pops, n_pops))
    indegree_matrix = np.zeros((n_pops, n_pops))
    effective_matrix = np.zeros((n_pops, n_pops))

    weight_mask = np.zeros((n_pops, n_pops), dtype=bool)
    indegree_mask = np.zeros((n_pops, n_pops), dtype=bool)
    effective_mask = np.zeros((n_pops, n_pops), dtype=bool)

    for i, tarKey in enumerate(pops):
        tarArea, tarPop = tarKey.split('-')
        for j, srcKey in enumerate(pops):
            srcArea, srcPop = srcKey.split('-')

            try:
                n_syn = synapse_number[tarArea][tarPop][srcArea][srcPop]
                w = abs(synapse_weight[tarArea][tarPop][srcArea][srcPop])
                n_neuron = neuron_number[tarArea][tarPop]

                if n_syn == 0 or n_neuron == 0:
                    raise ValueError

                indegree = n_syn / n_neuron
                weight_matrix[i, j] = abs(w)
                indegree_matrix[i, j] = indegree
                effective_matrix[i, j] = abs(w) * indegree / 1000.0

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


    # 归一化
    weight_norm = normalize(weight_matrix, weight_mask)
    indegree_norm = normalize(indegree_matrix, indegree_mask)
    effective_norm = normalize(effective_matrix, effective_mask)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    titles = ['Effective Weight (W × In-degree)', 'Synaptic Weight (W)', 'In-degree (Connections/Neuron)']
    matrices = [effective_matrix, weight_matrix, indegree_matrix]
    # matrices = [effective_norm, weight_norm, indegree_norm]
    raw_values = [effective_matrix, weight_matrix, indegree_matrix]
    masks = [effective_mask, weight_mask, indegree_mask]

    for ax, norm_mat, raw_mat, msk, ttl in zip(axes, matrices, raw_values, masks, titles):
        sns.heatmap(
            norm_mat,
            mask=msk,
            annot=np.round(raw_mat, 2),
            fmt=".2f",
            xticklabels=pops,
            yticklabels=pops,
            cmap='viridis',
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
    plt.savefig("connectivity_matrix.png")