import os
import sys
from config import expLIF_dict, input, layer_map, vis_content, \
    get_NN, get_SN, get_weight, get_weight_ext, externalRates, get_cc_delay, \
    getModelName, remove_dash_from_index_columns, get_ext_rate, net
from record import record_spike
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
NUM_THREADS_PER_SPIKE = 1
duration = 1000
buffer_size = 1

def split_indices(num_areas, num_gpus):
    # 平均分配索引到 num_gpus 个子列表
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_gpus - 1) // num_gpus  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus) if indices[i*chunk_size:(i+1)*chunk_size]]


def create_worker(i, idxs, area_list, model_i, NN, rate_ext, SN, weight, delay_cc, weight_ext):
    area = [area_list[j] for j in idxs]
    model_i, NeuronNumber_part, neuron_populations_part = create_model(
        model_i, area, NN, rate_ext, SN, weight, delay_cc, weight_ext
    )
    return i, (model_i, NeuronNumber_part, neuron_populations_part)

def simulation_worker(i, model_i, duration, neuron_pops_i, spike_data_i):
    model_i, spike_data_i = simulation(model_i, duration, neuron_pops_i, spike_data_i)
    return i, (model_i, spike_data_i)

def create_model(model, area_list, *args):
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
                        rate = rate_ext.loc[(area, layer, pop)] * 100
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
    print("Building Model of %u neurons and %u synapses of %u groups" % (total_neurons, total_synapses, syn_group_num))
    model.build()
    print("Loading Model")
    model.load(num_recording_timesteps=buffer_size)

    return model, NeuronNumber, neuron_populations

def simulation(model, duration, neuron_populations, spike_data):
    while model.t < duration:
        model.step_time()
        # model.pull_recording_buffers_from_device()
        # record_spike(neuron_populations, spike_data)
    return model, spike_data

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

    split_idx = split_indices(8,8)

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


    num_workers = 8
    NeuronNumber = [None] * num_workers
    neuron_populations = [None] * num_workers

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            f = executor.submit(create_worker, i, split_idx[i], area_list, model[i], NN, rate_ext, SN, weight, delay_cc, weight_ext)
            futures[f] = i

        for future in as_completed(futures):
            i = futures[future]
            try:
                i, (model_i, NeuronNumber_part, neuron_populations_part) = future.result()
            except Exception as e:
                import traceback
                print(f"Worker {i} raised exception:", e)
                traceback.print_exc()
                continue

            # 把线程返回的 GeNNModel 放回列表
            model[i] = model_i
            NeuronNumber[i] = NeuronNumber_part
            neuron_populations[i] = neuron_populations_part

    spike_data_blocks = [
        {
            area: {pop: [] for pop in neuron_populations[i][area].keys()}
            for area in neuron_populations[i].keys()
        }
        for i in range(num_workers)
    ]
    
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [
    #         executor.submit(simulation_worker, i, model[i],
    #                         neuron_populations[i], duration,
    #                         spike_data_blocks[i])
    #         for i in range(num_workers)
    #     ]

    #     for future in as_completed(futures):
    #         i, (model_i, spike_data_i) = future.result()
    #         model[i] = model_i
    #         spike_data_blocks[i] = spike_data_i

    for i in range(num_workers):
        model[i], spike_data_blocks[i] = simulation(model[i], duration, neuron_populations[i], spike_data_blocks[i])