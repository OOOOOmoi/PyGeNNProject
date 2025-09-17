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

def create_model(model, area_list, NeuronNumber, *args, **kwargs):
    nn = args[0]
    f_ext = args[1]
    sn = args[2]
    w = args[3]
    d_cc = args[4]
    if isinstance(area_list, str):
        area_list = [area_list]
    model.dt = 0.1
    layer_list = net["layer_list"]
    pop_list = net["population_list"]
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in args[0].index:
                    popName = area+pop+layer_map[layer]
                    popNum = nn.loc[(area, layer, pop)]
                    NeuronNumber[area][pop+layer_map[layer]] = popNum
                    if popNum != 0:
                        print("creating neuron group {popName} with {popNum} neurons".format(popName=popName, popNum=popNum))
                        if ("expLIF" in args):
                            neuronParam = expLIF_dict
                        else:
                            if (pop == "E"):
                                neuronParam = net['neuron_params_E']
                            else:
                                neuronParam = net['neuron_params_I']
                        if ("expLIF" in args):
                            params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                                        "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                                        "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                                        "TauRefrac": neuronParam['t_ref'], 
                                        "DeltaT": neuronParam['DeltaT'], "VT": neuronParam['VT']}
                        else:
                            params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                                        "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                                        "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                                        "TauRefrac": neuronParam['t_ref']}
                        if not args.poisson:
                            params["Ioffset"] = input[pop+layer_map[layer]] / 1000
                        if ("expLIF" in args):
                            neuron_pop = model.add_neuron_population(popName, popNum, expLIF_model, params, lif_init)
                        else:
                            neuron_pop = model.add_neuron_population(popName, popNum, "LIF", params, lif_init)
                        if args.poisson:
                            ext_weight = weight_ext.loc[(area, layer, pop)]
                            K = SN_ext.loc[(area, layer, pop)] / popNum
                            rate = externalRates(neuronParam, net_config['eta_ext'], K, ext_weight)
                            rate = rate_ext.loc[(area, layer, pop)]
                            # rate = 10*K
                            poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                            model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)

                        neuron_pop.spike_recording_enabled = True

                        total_neurons += popNum
                        neuron_populations[area][pop+layer_map[layer]] = neuron_pop

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

    spilt_idx = []


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
    idx = pd.IndexSlice
    NeuronNumber = defaultdict(dict)
    for i in range(8):
        area = area_list[spilt_idx[i]]
        nn = NN.loc[idx[area, :, :]]
        sn = SN.loc[idx[area, :, :], idx[area, :, :]]
        f_ext = rate_ext[idx[area, :, :]]
        w = weight.loc[idx[area, :, :], idx[area, :, :]]
        d_cc = delay_cc.loc[idx[area, area]]
        create_model(model[i], area_list, NeuronNumber, nn, f_ext, sn, w, d_cc)
