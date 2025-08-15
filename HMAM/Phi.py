from config import data_dir, layer_map, vis_content, get_NN, get_SN, get_weight, get_weight_ext, externalRates
from config import net as net_config
import os
import itertools as product
import pandas as pd
from sigert import mu_sigma
from theory_helpers import nu0_fb

def getMatrix():
    NN = get_NN()
    SN, SN_ext = get_SN()
    weight, weight_sd = get_weight()
    weight_ext, weight_ext_sd = get_weight_ext()
    area_list = net_config['area_list'][30]
    layer_list = net_config['layer_list']
    pop_list = net_config['population_list']
    
    # 增加 ext 这个来源
    all_pops = list(product.product(area_list, layer_list, pop_list))
    all_pops_with_ext = all_pops + ['ext']

    # 初始化 DataFrame
    weight_matrix = pd.DataFrame(index=all_pops, columns=all_pops_with_ext, dtype=float)
    indegree_matrix = pd.DataFrame(index=all_pops, columns=all_pops_with_ext, dtype=float)

    # 普通 tar-src
    for tar_area, src_area in product.product(area_list, area_list):
        for tar_layer, src_layer in product.product(layer_list, layer_list):
            for tar_pop, src_pop in product.product(pop_list, pop_list):
                tar = (tar_area, tar_layer, tar_pop)
                src = (src_area, src_layer, src_pop)
                if tar in SN.index and src in SN.columns:
                    synNum = SN.loc[tar, src]
                    wAve = weight.loc[tar, src]
                    indegree_matrix.loc[tar, src] = synNum / NN.loc[tar]
                    weight_matrix.loc[tar, src] = wAve

    # ext 列（外部输入）
    for tar_area, tar_layer, tar_pop in all_pops:
        tar = (tar_area, tar_layer, tar_pop)
        popNum = NN.loc[tar]
        ext_weight_val = weight_ext.loc[tar]
        K_ext = SN_ext.loc[tar] / popNum
        indegree_matrix.loc[tar, 'ext'] = K_ext
        weight_matrix.loc[tar, 'ext'] = ext_weight_val

    # rate_ext_matrix
    rate_ext_matrix = pd.DataFrame(index=all_pops, dtype=float)
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if pop == "E":
                    neuronParam = net_config['neuron_params_E']
                else:
                    neuronParam = net_config['neuron_params_I']
                popNum = NN.loc[(area, layer, pop)]
                ext_weight_val = weight_ext.loc[(area, layer, pop)]
                K_ext = SN_ext.loc[(area, layer, pop)] / popNum
                rate_ext_matrix.loc[(area, layer, pop)] = externalRates(
                    neuronParam, net_config['eta_ext'], K_ext, ext_weight_val
                )
    
    return indegree_matrix, weight_matrix, rate_ext_matrix


def Phi(rate, K_matrix, J_matrix, rate_ext, SingleNeuronDict):
    mu, sigma = mu_sigma(rate, K_matrix, J_matrix, rate_ext, SingleNeuronDict)
    NP = SingleNeuronDict
    return list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                    1.e-3 * NP['tau_m'],
                                                    1.e-3 * NP['tau_syn_ex'],
                                                    1.e-3 * NP['t_ref'],
                                                    NP['V_th'] - NP['E_L'],
                                                    NP['V_reset'] - NP['E_L']),
                        mu, sigma))

def fp_solve(Phi, rate, K_matrix, J_matrix, rate_ext, SingleNeuronDict):
    def f():
        return Phi(rate)-rate
    result = optimize.fsolve(f, rates_init, full_output=1)
    
