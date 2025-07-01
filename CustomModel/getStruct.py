from config import collection_params
from itertools import product
from nested_dict import nested_dict
from scipy.stats import norm

def parse_specific_scale_syn(config_str):
    specific_scale_syn = {}
    lines = config_str.strip().splitlines()
    
    for line in lines:
        tarArea, tarPop, srcArea, srcPop, value = line.strip().split(',')
        value = float(value)

        # 构建嵌套结构
        specific_scale_syn.setdefault(tarArea, {})
        specific_scale_syn[tarArea].setdefault(tarPop, {})
        specific_scale_syn[tarArea][tarPop].setdefault(srcArea, {})
        specific_scale_syn[tarArea][tarPop][srcArea][srcPop] = value

    return specific_scale_syn

def has_key_path(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return False
    return True

def get_struct():
    all_layers = ['1', '23', '4', '5', '6']
    model_content = collection_params['model_content']
    cell_types_by_layer = {
        '1': ['H'],
        '23': ['E', 'S', 'P', 'V'],
        '4': ['E', 'S', 'P', 'V'],
        '5': ['E', 'S', 'P', 'V'],
        '6': ['E', 'S', 'P', 'V']
    }

    model_structure = {}

    for area, layers in model_content.items():
        # 如果层列表为空，则表示使用所有层
        selected_layers = all_layers if not layers else layers
        model_structure[area] = {}

        pops = []
        for layer in selected_layers:
            pops.extend([ctype + layer for ctype in cell_types_by_layer[layer]])
        model_structure[area] = pops

    return model_structure

def get_weight_factor():
    single_neuron_dict=collection_params['single_neuron_dict']
    C_m_E = single_neuron_dict['C_m']
    tau_m_E = single_neuron_dict['tau_m']
    tau_syn_E = single_neuron_dict['tau_syn']
    PSC_over_PSP = ((C_m_E**(-1) * tau_m_E * tau_syn_E / (tau_syn_E - tau_m_E) *
                    ((tau_m_E / tau_syn_E) ** (- tau_m_E / (tau_m_E - tau_syn_E)) -
                    (tau_m_E / tau_syn_E) ** (- tau_syn_E / (tau_m_E - tau_syn_E)))) ** (-1))
    return PSC_over_PSP

def getWeightMap(structure):
    PSC_over_PSP= get_weight_factor()
    SynapsesWeightMean=nested_dict()
    SynapsesWeightSd=nested_dict()
    connection_params=collection_params['connection_params']
    alpha_norm = connection_params['alpha_norm']
    beta_norm = connection_params['beta_norm']
    specific_scale_syn = parse_specific_scale_syn(collection_params['specific_scale_syn'])
    for tarArea, tarList in structure.items():
        for srcArea, srcList in structure.items():
            for tarPop, srcPop in product(tarList, srcList):
                if tarArea == srcArea:
                    if srcPop[0] == 'E':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * alpha_norm[srcPop]*beta_norm[tarPop] * connection_params['PSP_e']
                    if srcPop[0] == 'H':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['g_H'] * alpha_norm[srcPop] * connection_params['PSP_e']
                    if srcPop[0] == 'P':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['g_P'] * alpha_norm[srcPop] * connection_params['PSP_e']
                    if srcPop[0] == "S":
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['g_S'] * alpha_norm[srcPop] * connection_params['PSP_e']
                    if srcPop[0] == "V":
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['g_V'] * alpha_norm[srcPop] * connection_params['PSP_e']
                    if tarPop == "H1" and srcPop[0] != "E":
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= 0.5
                    SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] = abs(SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop]) * connection_params['PSC_rel_sd_normal']
                    if tarPop == 'E23' and srcPop == 'E4':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4'] * connection_params['PSC_rel_sd_normal']
                else:
                    if tarPop[0] == 'E':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_factor']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_factor']
                    else:
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_I_factor']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_I_factor']
                if has_key_path(specific_scale_syn, tarArea, tarPop, srcArea, srcPop):
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= specific_scale_syn[tarArea][tarPop][srcArea][srcPop]
                    SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= specific_scale_syn[tarArea][tarPop][srcArea][srcPop]
    return SynapsesWeightMean.to_dict(), SynapsesWeightSd.to_dict()

def getDelayMap(structure, Dist):
    type_name = ["E", "I"]

    mean_delay = {"E": 1.5, "I": 0.75}
    delay_sd = {"E": 0.75, "I": 0.375}
    speed = 3.5#mm/s
    delay_rel = 0.5
    max_intra_area_delay = 0
    quantile = 0.9999
    normal_quantile_cdf = norm.ppf(quantile)
    max_delay = {type: mean_delay[type] + (delay_sd[type] * normal_quantile_cdf)
                    for type in type_name}
    delayMap=nested_dict()
    for tarArea, tarList in structure.items():
        for srcArea, srcList in structure.items():
            for tarPop, srcPop in product(tarList, srcList):
                if tarArea == srcArea:
                    if srcPop.startswith("E"):
                        meanDelay = mean_delay["E"]
                        sd = delay_sd["E"]
                        max_d = max_delay["E"]
                    else:
                        meanDelay = mean_delay["I"]
                        sd = delay_sd["I"]
                        max_d = max_delay["I"]
                else:
                    meanDelay = Dist[tarArea][srcArea]/speed
                    sd = meanDelay*delay_rel
                    max_d = max(max_intra_area_delay, meanDelay + (sd * normal_quantile_cdf))
                delayMap[tarArea][tarPop][srcArea][srcPop]={'ave':meanDelay,'sd':sd,'max':max_d}
    return delayMap
