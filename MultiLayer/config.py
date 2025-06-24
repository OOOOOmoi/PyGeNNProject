import numpy as np
from itertools import product
from nested_dict import nested_dict
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import json
# from MultiLayer import PopList, TYPE_NAMES
LayerList = ["4"]
Area = "V1"
TYPE_NAMES = ["E", "S", "P", "V"]
connection_params = {
    # Relative inhibitory synaptic strength (in relative units).
    'g': -16.,
    'g_H': -2.,
    'g_V': -2.,
    'g_P': -2.,
    'g_S': -2.,
    
    'alpha_norm': {
          'H1':  1,
          'E23': 1,'S23': 1,'V23': 1,'P23': 1,  
          'E4':  1,'S4':  1/2,'V4':  1/2,'P4':  1/2,
          'E5':  1,'S5':  1,'V5':  1,'P5':  1,
          'E6':  1,'S6':  1,'V6':  1,'P6':  1,
        },
    'beta_norm':{"H1" : 3.9,
                 "E23" : 0.71, "S23" : 1., "P23" : 0.48, "V23" :0.9,
                 "E4" : 1.66/10, "S4" : 0.24, "P4" : 0.8, "V4" : 0.46,
                 "E5" : 0.95, "S5" : 0.48, "P5" :1.09, "V5" : 1.2,
                 "E6" : 1.12, "S6" : 0.63, "P6" : 0.42, "V6" : 0.5,},

    'input':{
        "H1": 501.0 + -30.0,
        "E23": 501.0 + 70.0, "S23": 501.0 + -7.0,  "P23": 501.0 + -6.0,  "V23": 501.0 + -6.0,
        "E4": 501.0 + 50.0,  "S4": 501.0 + -0.0,   "P4": 501.0 + 10.0,   "V4": 501.0 + 10.0,
        "E5": 501.0 + 50.0,  "S5": 501.0 + 10.0,   "P5": 501.0 + -10.0,  "V5": 501.0 - 20.0,
        "E6": 501.0 + 50.0,  "S6": 501.0 + 0.0,    "P6": 501.0 + 0.0,    "V6": 501.0 - 10.0,
    },

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight 
    'PSP_e': 0.15, # mV
    'PSP_e_23_4': 0.3, #mV
    'PSP_e_5_h1': 0.15, #mV
    # synaptic weight  for external input
    'PSP_ext': 0.15, #mV

    # relative SD of normally distributed synaptic weights
    'PSC_rel_sd_normal': 0.1,
    # relative SD of lognormally distributed synaptic weights
    'PSC_rel_sd_lognormal': 3.0,
}


C_m_E = 500
tau_m_E = 20
tau_syn_E = 0.5
PSC_over_PSP = ((C_m_E**(-1) * tau_m_E * tau_syn_E / (tau_syn_E - tau_m_E) *
                  ((tau_m_E / tau_syn_E) ** (- tau_m_E / (tau_m_E - tau_syn_E)) -
                  (tau_m_E / tau_syn_E) ** (- tau_syn_E / (tau_m_E - tau_syn_E)))) ** (-1))

SynapsesWeightMean=nested_dict()
SynapsesWeightSd=nested_dict()
alpha_norm = connection_params['alpha_norm']
beta_norm = connection_params['beta_norm']
def getWeightMap():
    for layerTar, layerSrc in product(LayerList, LayerList):
        for typeTar, typeSrc in product(TYPE_NAMES, TYPE_NAMES):
            if layerSrc == "1":
                typeSrc = "H"
            if layerTar == "1":
                typeTar = "H"
            srcPop = typeSrc + layerSrc
            tarPop = typeTar + layerTar
            if typeSrc == 'E':
                SynapsesWeightMean[tarPop][srcPop] = PSC_over_PSP * alpha_norm[srcPop]*beta_norm[tarPop] * connection_params['PSP_e']
            if typeSrc == 'H':
                SynapsesWeightMean[tarPop][srcPop] = PSC_over_PSP * connection_params['g_H'] * alpha_norm[srcPop] * connection_params['PSP_e']
            if typeSrc == 'P':
                SynapsesWeightMean[tarPop][srcPop] = PSC_over_PSP * connection_params['g_P'] * alpha_norm[srcPop] * connection_params['PSP_e']
            if typeSrc == "S":
                SynapsesWeightMean[tarPop][srcPop] = PSC_over_PSP * connection_params['g_S'] * alpha_norm[srcPop] * connection_params['PSP_e']
            if typeSrc == "V":
                SynapsesWeightMean[tarPop][srcPop] = PSC_over_PSP * connection_params['g_V'] * alpha_norm[srcPop] * connection_params['PSP_e']
            if tarPop == "H1" and typeSrc != "E":
                SynapsesWeightMean[tarPop][srcPop] *= 0.5
            SynapsesWeightSd[tarPop][srcPop] = abs(SynapsesWeightMean[tarPop][srcPop]) * connection_params['PSC_rel_sd_normal']
            if tarPop == 'E23' and srcPop == 'E4':
                SynapsesWeightMean[tarPop][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4']
                SynapsesWeightSd[tarPop][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4'] * connection_params['PSC_rel_sd_normal']
    return SynapsesWeightMean.to_dict(), SynapsesWeightSd.to_dict()

def plot_effective_weight_heatmap(SynapsesWeightMean, SynapsesNumber, NeuronNumber, areaName, title='Synapse Connectivity Maps'):
    target_pops = list(SynapsesWeightMean.keys())
    source_pops = sorted({src for tar in SynapsesWeightMean.values() for src in tar})

    n_tar = len(target_pops)
    n_src = len(source_pops)

    weight_matrix = np.zeros((n_tar, n_src))
    in_deg_matrix = np.zeros((n_tar, n_src))
    effective_weight_matrix = np.zeros((n_tar, n_src))

    weight_mask = np.zeros((n_tar, n_src), dtype=bool)
    in_deg_mask = np.zeros((n_tar, n_src), dtype=bool)
    effective_mask = np.zeros((n_tar, n_src), dtype=bool)

    for i, tarPop in enumerate(target_pops):
        for j, srcPop in enumerate(source_pops):
            try:
                syn_num = SynapsesNumber[areaName][tarPop][areaName][srcPop]
                neuron_num = NeuronNumber[areaName][tarPop]
                weight = abs(SynapsesWeightMean[tarPop][srcPop])

                if syn_num == 0 or neuron_num == 0:
                    raise ValueError

                in_deg = syn_num / neuron_num
                in_deg_matrix[i, j] = in_deg
                weight_matrix[i, j] = weight
                effective_weight_matrix[i, j] = weight * in_deg

            except:
                weight_mask[i, j] = True
                in_deg_mask[i, j] = True
                effective_mask[i, j] = True

    def normalize(mat, mask):
        m = np.copy(mat)
        if np.any(~mask):
            max_val = np.max(m[~mask])
            if max_val > 0:
                m = m / max_val
        return m

    W = solve_weights_with_signs(in_deg_matrix, source_pops)
    weight_dict = {pop: float(val) for pop, val in zip(source_pops, W)}
    filename = "Wsolution.json"
    with open(filename, 'w') as f:
        json.dump(weight_dict, f, indent=4)
    # weight_matrix_norm = normalize(weight_matrix, weight_mask)
    # in_deg_matrix_norm = normalize(in_deg_matrix, in_deg_mask)
    effective_weight_matrix_norm = normalize(effective_weight_matrix, effective_mask)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    titles = ['Effective Weight (W × In-degree)', 'Raw Synaptic Weight (W)', 'In-degree (Connections/Neuron)']
    matrices = [effective_weight_matrix_norm, weight_matrix, in_deg_matrix]
    masks = [effective_mask, weight_mask, in_deg_mask]

    for ax, mat, msk, ttl in zip(axes, matrices, masks, titles):
        sns.heatmap(
            mat,
            mask=msk,
            annot=True,
            fmt=".2f",  # 两位小数
            xticklabels=source_pops,
            yticklabels=target_pops,
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

    plt.suptitle(title, fontsize=18)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('connectionMap.png')


def solve_weights_with_signs(ind_matrix, source_pops):
    """
    求解权重向量 w，使得 ind_matrix @ w ≈ 0，满足：
    - w[j] > 0 if source_pops[j] is excitatory
    - w[j] < 0 if source_pops[j] is inhibitory
    """

    n_sources = len(source_pops)

    # 设置符号约束
    lower_bounds = np.zeros(n_sources)
    upper_bounds = np.zeros(n_sources)

    for i, pop in enumerate(source_pops):
        if pop.startswith("E"):  # 兴奋性：权重大于0
            lower_bounds[i] = 1e-12
            upper_bounds[i] = np.inf
        else:  # 抑制性：权重小于0
            lower_bounds[i] = -np.inf
            upper_bounds[i] = -1e-12

    # 使用最小二乘求解，满足 ind_matrix @ w ≈ 0 且符号约束
    result = lsq_linear(
        ind_matrix, np.zeros(ind_matrix.shape[0]),
        bounds=(lower_bounds, upper_bounds),
        lsmr_tol='auto',
        verbose=1
    )

    if result.success:
        return result.x
    else:
        raise ValueError("解失败，可能没有满足所有约束条件的解。")