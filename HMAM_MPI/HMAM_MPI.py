from HMAM.config import expLIF_dict, input, layer_map, vis_content, \
    get_NN, get_SN, get_weight, get_weight_ext, externalRates, get_cc_delay, \
    getModelName, remove_dash_from_index_columns, get_ext_rate, net
import os
import sys
import numpy as np
from argparse import ArgumentParser, Namespace
import string
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from pygenn.cuda_backend import DeviceSelect
from time import perf_counter
from itertools import product
import pandas as pd
from collections import defaultdict
from nested_dict import nested_dict
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
NUM_THREADS_PER_SPIKE = 1
duration = 1000

def worker(i, idxs, area_list, model_i, NN, rate_ext, SN, weight, delay_cc, weight_ext):
    # 每个 worker 负责一部分区域
    area = [area_list[j] for j in idxs]
    return i, create_model(model_i, area, defaultdict(dict), defaultdict(dict),
                           NN, rate_ext, SN, weight, delay_cc, weight_ext)

def create_model(model, area_list, NeuronNumber, neuron_populations, *args, **kwargs):
    
    NN = args[0]
    rate_ext = args[1]
    SN = args[2]
    weight = args[3]
    delay_cc = args[4]
    weight_ext = args[5]
    if isinstance(area_list, str):
        area_list = [area_list]
    model.dt = 0.1
    layer_list = net["layer_list"]
    pop_list = net["population_list"]
    lif_init = {"V": init_var("Uniform", {"max": -50.0, "min": -200.0}), "RefracTime": 0.0}
    poisson_init = {"current": 0.0}
    total_neurons = 0
    NeuronNumber = defaultdict(dict)
    neuron_populations = defaultdict(dict)
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in args[0].index:
                    popName = area+pop+layer_map[layer]
                    popNum = NN.loc[(area, layer, pop)]
                    NeuronNumber[area][pop+layer_map[layer]] = popNum
                    if popNum != 0:
                        print("creating neuron group {popName} with {popNum} neurons".format(popName=popName, popNum=popNum))
                        if (pop == "E"):
                            neuronParam = net['neuron_params_E']
                        else:
                            neuronParam = net['neuron_params_I']
                        params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                                    "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                                    "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                                    "TauRefrac": neuronParam['t_ref']}
                        neuron_pop = model.add_neuron_population(popName, popNum, "LIF", params, lif_init)
                        ext_weight = weight_ext.loc[(area, layer, pop)]
                        rate = rate_ext.loc[(area, layer, pop)]
                        # rate = 10*K
                        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                        model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)

                        neuron_pop.spike_recording_enabled = True

                        total_neurons += popNum
                        neuron_populations[area][pop+layer_map[layer]] = neuron_pop

    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 2})
    inh_curr_init = init_postsynaptic("ExpCurr", {"tau": 5})
    total_synapses = 0
    syn_group_num = 0
    for tar_area, src_area in product(area_list, area_list):
        for tar_layer, src_layer in product(layer_list, layer_list):
            for tar_pop, src_pop in product(pop_list, pop_list):
                tar = (tar_area, tar_layer, tar_pop)
                src = (src_area, src_layer, src_pop)
                if tar in SN.index and src in SN.columns:
                    tarName = tar_area+tar_pop+layer_map[tar_layer]
                    srcName = src_area+src_pop+layer_map[src_layer]
                    synName = srcName + "_to_" + tarName
                    synNum = SN.loc[tar, src]
                    wAve = weight.loc[tar, src] / 1000
                    wSd = wAve / 10 / 1000
                    if src_area == tar_area:
                        if src_pop == 'E':
                            meanDelay = net['delay_e']
                            delay_sd = net['delay_e_sd']
                        else:
                            meanDelay = net['delay_i']
                            delay_sd = net['delay_i_sd']
                    else:
                        meanDelay = delay_cc.loc[(src_area, tar_area)]
                        delay_sd = delay_cc_sd.loc[(src_area, tar_area)]
                    if synNum > 0:
                        tarPop = neuron_populations[tar_area][tar_pop+layer_map[tar_layer]]
                        srcPop = neuron_populations[src_area][src_pop+layer_map[src_layer]]
                        quantile = 0.9999
                        normal_quantile_cdf = norm.ppf(quantile)
                        max_delay = meanDelay + (delay_sd * normal_quantile_cdf)
                        connect_params = {"num": synNum}
                        # Build distribution for delay parameters
                        d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_delay}
                        total_synapses += synNum
                        syn_group_num += 1
                        # Build unique synapse name
                        matrix_type = "PROCEDURAL"
                        if src_pop == 'E':
                            curr_init = exp_curr_init
                            w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                        else:
                            curr_init = inh_curr_init
                            w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                        static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                            {"g": init_var("NormalClipped", w_dist),
                                                            "d": init_var("NormalClippedDelay", d_dist)})
                        syn_pop = model.add_synapse_population(synName, matrix_type,
                        srcPop, tarPop,
                        static_synapse_init, curr_init,
                        init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                        
                        syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / model.dt))

                        if matrix_type=="PROCEDURAL":
                            syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
    model.build()
    model.load(num_recording_timesteps=100)


    return model, NeuronNumber, neuron_populations

if __name__ == '__main__':
    model = []
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device0", device_select_method=DeviceSelect.MANUAL, manual_device_id=0))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device1", device_select_method=DeviceSelect.MANUAL, manual_device_id=1))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device2", device_select_method=DeviceSelect.MANUAL, manual_device_id=2))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device3", device_select_method=DeviceSelect.MANUAL, manual_device_id=3))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device4", device_select_method=DeviceSelect.MANUAL, manual_device_id=4))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device5", device_select_method=DeviceSelect.MANUAL, manual_device_id=5))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device6", device_select_method=DeviceSelect.MANUAL, manual_device_id=6))
    model.append(GeNNModel("float", "HMAM_MPI_CODE/model_on_device7", device_select_method=DeviceSelect.MANUAL, manual_device_id=7))
    area_list = net["area_list"]
    area_list = [s.replace("-", "") for s in area_list]

    spilt_idx = [[1,2],
                 [3,4],
                 [5,6],
                 [7,8],
                 [9,10],
                 [11,12],
                 [13,14],
                 [15,16]]


    NN=get_NN()
    NN = remove_dash_from_index_columns(NN)
    SN, SN_ext = get_SN()
    SN = remove_dash_from_index_columns(SN)
    SN_ext = remove_dash_from_index_columns(SN_ext)
    rate_ext = get_ext_rate()
    rate_ext = remove_dash_from_index_columns(rate_ext)
    weight, weight_sd = get_weight()
    weight = remove_dash_from_index_columns(weight)
    weight_sd = remove_dash_from_index_columns(weight_sd)
    delay_cc, delay_cc_sd = get_cc_delay()
    delay_cc = remove_dash_from_index_columns(delay_cc)
    delay_cc_sd = remove_dash_from_index_columns(delay_cc_sd)
    weight_ext, weight_ext_sd = get_weight_ext()
    weight_ext = remove_dash_from_index_columns(weight_ext)
    weight_ext_sd = remove_dash_from_index_columns(weight_ext_sd)
    idx = pd.IndexSlice
    for i in range(8):
        idxs = spilt_idx[i]
        area = [area_list[j] for j in idxs]
        model[i], NeuronNumber, neuron_populations = create_model(model[i], area, NN, rate_ext, SN, weight, delay_cc, weight_ext)
