from config import collection_params
from itertools import product
from nested_dict import nested_dict
from scipy.stats import norm
import os
import pandas as pd

def generate_scale_excel_from_structure(area, filename='specific_scale_syn.xlsx'):
    """
    从 structure[area] 的 neuron populations 生成或更新 Excel 表格。
    - 行是 target population（tarPop）
    - 列是 source population（srcPop）
    - 保留已有值，仅补充新结构
    """
    structure = get_struct()  # 获取当前的建模结构
    pops = structure[area]  # 当前建模的所有 population 名称

    if os.path.exists(filename):
        old_df = pd.read_excel(filename, index_col=0)
    else:
        old_df = pd.DataFrame()

    # 创建一个全新的表格，包含当前 pops 的完整组合
    new_df = pd.DataFrame(index=pops, columns=pops)

    # 保留旧表格中已有的匹配项值（交叉引用）
    for tar in pops:
        for src in pops:
            if tar in old_df.index and src in old_df.columns:
                new_df.loc[tar, src] = old_df.loc[tar, src]

    new_df.to_excel(filename)
    print(f"Excel 文件已生成（保留旧值，补充新结构）: {filename}")


def load_scale_dict_from_excel(area='V1', filename='specific_scale_syn.xlsx'):
    """
    从 Excel 表格中读取缩放因子，生成嵌套结构的字典：
    dict[tarArea][tarPop][srcArea][srcPop] = value
    """
    df = pd.read_excel(filename, index_col=0)
    scale_dict = nested_dict()

    for tarPop in df.index:
        for srcPop in df.columns:
            val = df.at[tarPop, srcPop]
            if pd.notna(val):  # 非空有效
                val = float(val)
                # 构造嵌套字典
                scale_dict[tarPop][srcPop] = val

    return scale_dict


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
    single_neuron_dict=collection_params['single_neuron_dict_of_weight']
    PSC_over_PSP = nested_dict()
    for type_, params in single_neuron_dict.items():
        C_m = params['C_m']
        tau_m = params['tau_m']
        tau_syn = params['tau_syn']
        PSC_over_PSP[type_] = ((C_m**(-1) * tau_m * tau_syn / (tau_syn - tau_m) *
                        ((tau_m / tau_syn) ** (- tau_m / (tau_m - tau_syn)) -
                        (tau_m / tau_syn) ** (- tau_syn / (tau_m - tau_syn)))) ** (-1))
    return PSC_over_PSP

def getWeightMap(structure, args):
    PSC_over_PSP_= get_weight_factor()
    SynapsesWeightMean=nested_dict()
    SynapsesWeightSd=nested_dict()
    connection_params=collection_params['connection_params']
    alpha_norm = connection_params['alpha_norm']
    beta_norm = connection_params['beta_norm']
    specific_scale_syn = load_scale_dict_from_excel()
    if "free_scale_syn" in args:
        specific_scale_syn ['V5']['E5'] += float(args.free_scale_syn)
    for tarArea, tarList in structure.items():
        for srcArea, srcList in structure.items():
            for tarPop, srcPop in product(tarList, srcList):
                type_ = srcPop[0]
                PSC_over_PSP = PSC_over_PSP_[type_]
                if tarArea == srcArea:
                    if srcPop[0] == 'E':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * alpha_norm[srcPop] * connection_params['PSP_e']
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
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4'] / 4
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4'] * connection_params['PSC_rel_sd_normal']
                else:
                    if tarPop[0] == 'E':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_factor']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_factor']
                    else:
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_I_factor']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_I_factor']
                if has_key_path(specific_scale_syn, tarPop, srcPop):
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= specific_scale_syn[tarPop][srcPop]
                    SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= specific_scale_syn[tarPop][srcPop]
                SynapsesWeightMean[tarArea][tarPop]['external']['external'] = connection_params['PSP_ext'] * PSC_over_PSP_['E']
    return SynapsesWeightMean.to_dict(), SynapsesWeightSd.to_dict()

def getWeightMap_full_type(structure, args):
    Cm = collection_params['single_neuron_dict']['Cm']
    gL = collection_params['single_neuron_dict']['gL']
    connection_params=collection_params['connection_params']
    beta_norm = connection_params['beta_norm']
    PSC_over_PSP_ = nested_dict()
    for type_, value_ in Cm.items():
        C_m = Cm[type_]
        tau_m = C_m / gL[type_]
        tau_syn = 0.5
        PSC_over_PSP_[type_] = ((C_m**(-1) * tau_m * tau_syn / (tau_syn - tau_m) *
                        ((tau_m / tau_syn) ** (- tau_m / (tau_m - tau_syn)) -
                        (tau_m / tau_syn) ** (- tau_syn / (tau_m - tau_syn)))) ** (-1))
    SynapsesWeightMean=nested_dict()
    SynapsesWeightSd=nested_dict()
    for tarArea, tarList in structure.items():
        for srcArea, srcList in structure.items():
            for tarPop, srcPop in product(tarList, srcList):
                type_ = tarPop
                PSC_over_PSP = PSC_over_PSP_[type_]
                if srcPop[0] == 'E':
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP
                if srcPop[0] == 'H':
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * -1
                if srcPop[0] == 'P':
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * -1
                if srcPop[0] == "S":
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * -1
                if srcPop[0] == "V":
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * -1
                # if tarPop =='S4' and srcPop[0] == 'E':
                #         SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= beta_norm[tarPop]
                # if tarPop =='S5' and srcPop[0] == 'E':
                #     SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= beta_norm[tarPop]
                if tarPop =='S4' and srcPop == tarPop:
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= 9
                if tarPop =='S5' and srcPop == tarPop:
                    SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= 10
                # if tarPop == "H1" and srcPop[0] != "E":
                #     SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= 0.5
                SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] = abs(SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop]) * connection_params['PSC_rel_sd_normal']
                # if tarPop == 'E23' and srcPop == 'E4':
                #     SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4'] / 4
                #     SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] = PSC_over_PSP * connection_params['PSP_e_23_4'] * connection_params['PSC_rel_sd_normal']
                if tarArea != srcArea:
                    if tarPop[0] == 'E':
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_factor']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_factor']
                    else:
                        SynapsesWeightMean[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_I_factor']
                        SynapsesWeightSd[tarArea][tarPop][srcArea][srcPop] *= connection_params['cc_weights_I_factor']
                SynapsesWeightMean[tarArea][tarPop]['external']['external'] = PSC_over_PSP_[type_]
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
