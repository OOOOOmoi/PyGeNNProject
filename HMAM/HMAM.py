import numpy as np
from argparse import ArgumentParser, Namespace
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from pygenn.cuda_backend import DeviceSelect
from time import perf_counter
from itertools import product
import os
import json
import random
import string
import matplotlib.pyplot as plt
from collections import defaultdict
from nested_dict import nested_dict
from config import net as net_config
from config import data_dir, layer_map, vis_content, get_NN, get_SN, get_weight, get_weight_ext, externalRates
from scipy.stats import norm
from record import record_spike, save_spike, record_inSyn, save_inSyn
from visual import visualize, generate_unique_suffix

DT_MS=0.1
NUM_THREADS_PER_SPIKE=8

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--duration", type=float, default=1000.0, nargs="?", help="Duration to simulate (ms)")
    parser.add_argument("--stim", action="store_true", help="Whether to apply a stimulus")
    parser.add_argument("--stim-start", type=float, default=300, help="start time of stim")
    parser.add_argument("--stim-end", type=float, default=800, help="end time of stim")
    parser.add_argument("--buffer", action="store_true", help="Whether use buffer store spike")
    parser.add_argument("--buffer-size", type=int, default=100, nargs="?", help="Size of recording buffer")
    parser.add_argument("--SPARSE", action="store_true", help="Whether use sparse connectivity")
    parser.add_argument("--inSyn", action="store_true", help="Whether record inSyn")
    parser.add_argument("--save-spike", action="store_true", help="whether store spike")
    parser.add_argument("--device", type=int, default=0, help="Device ID to use for simulation")
    parser.add_argument("--poisson", action="store_true", help="Whether use poisson input")
    return parser

def parse_all_args():
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # 手动解析未知参数（--key val 或 --flag）
    extra_args = {}
    key = None
    for item in unknown:
        if item.startswith('--'):
            key = item.lstrip('--').replace('-', '_')
            extra_args[key] = True  # 默认是布尔开关
        elif key is not None:
            extra_args[key] = item  # 赋值
            key = None

    # 合并已知参数与未知参数
    args_dict = vars(args)
    args_dict.update(extra_args)
    return Namespace(**args_dict)

if __name__ == "__main__":
    args = parse_all_args()
    model_name = 'HMAM'
    model = GeNNModel("float", model_name)
    model.dt = 0.1
    model.fuse_postsynaptic_models = not args.inSyn
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE

    suffix = generate_unique_suffix()

    area_list = net_config['area_list'][30]
    if isinstance(area_list, str):
        area_list = [area_list]
    layer_list = net_config['layer_list']
    pop_list = net_config['population_list']


    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})

    trigger_pulse_model = pygenn.create_current_source_model(
        "trigger_pulse",
        params=["start_time","end_time","magnitude"],  # 参数：噪声强度
        injection_code=
        """
        if (t >= start_time && t < end_time) {
            injectCurrent(magnitude);
        }
        """
    )

    NN=get_NN()
    SN, SN_ext = get_SN()
    total_neurons = 0
    weight_ext, weight_ext_sd = get_weight_ext()
    neuron_populations = defaultdict(dict)
    NeuronNumber = defaultdict(dict)
    poisson_init = {"current": 0.0}
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": 0.0}
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in NN.index:
                    popName = area+pop+layer_map[layer]
                    popNum = NN.loc[(area, layer, pop)]
                    print("creating neuron group {popName} with {popNum} neurons".format(popName=popName, popNum=popNum))
                    if (pop == "E"):
                        neuronParam = net_config['neuron_params_E']
                    else:
                        neuronParam = net_config['neuron_params_I']
                    params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                                "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                                "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                                "TauRefrac": neuronParam['t_ref']}
                    neuron_pop = model.add_neuron_population(popName, popNum, "LIF", params, lif_init)
                    if args.poisson:
                        ext_weight = weight_ext.loc[(area, layer, pop)]
                        K = SN_ext.loc[(area, layer, pop)] / popNum
                        rate = externalRates(neuronParam, net_config['eta_ext'], K, ext_weight)
                        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                        model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)
                    else:
                        params["Ioffset"] = 0.551
                    neuron_pop.spike_recording_enabled = True

                    total_neurons += popNum
                    NeuronNumber[area][pop+layer_map[layer]] = popNum
                    neuron_populations[area][pop+layer_map[layer]] = neuron_pop
                    
    total_synapses = 0
    synapse_populations = nested_dict()
    weight, weight_sd = get_weight()
    for tar_area, src_area in product(area_list, area_list):
        for tar_layer, src_layer in product(layer_list, layer_list):
            for tar_pop, src_pop in product(pop_list, pop_list):
                tar = (tar_area, tar_layer, tar_pop)
                src = (src_area, src_layer, src_pop)
                if tar in SN.index and src in SN.columns:
                    tarName = tar_area+tar_pop+layer_map[tar_layer]
                    srcName = src_area+src_pop+layer_map[src_layer]
                    synName = srcName + "_to_" + tarName
                    tarPop = neuron_populations[tar_area][tar_pop+layer_map[tar_layer]]
                    srcPop = neuron_populations[src_area][src_pop+layer_map[src_layer]]
                    synNum = SN.loc[tar, src]
                    wAve = weight.loc[tar, src]
                    wSd = weight_sd.loc[tar, src]
                    if src_pop == 'E':
                        meanDelay = net_config['delay_e']
                        delay_sd = net_config['delay_e_sd']
                    else:
                        meanDelay = net_config['delay_i']
                        delay_sd = net_config['delay_i_sd']
                    if synNum > 0:
                        quantile = 0.9999
                        normal_quantile_cdf = norm.ppf(quantile)
                        max_delay = meanDelay + (delay_sd * normal_quantile_cdf)
                        connect_params = {"num": synNum}
                        # Build distribution for delay parameters
                        d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_delay}
                        total_synapses += synNum
                        # Build unique synapse name
                        matrix_type = "SPARSE" if args.SPARSE else "PROCEDURAL"
                        if src_pop == 'E':
                            w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                        else:
                            w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                        static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                            {"g": init_var("NormalClipped", w_dist),
                                                            "d": init_var("NormalClippedDelay", d_dist)})
                        syn_pop = model.add_synapse_population(synName, matrix_type,
                        srcPop, tarPop,
                        static_synapse_init, exp_curr_init,
                        init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                        
                        syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / DT_MS))

                        if matrix_type=="PROCEDURAL":
                            syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
                        synapse_populations[tar_area][tar_pop+layer_map[tar_layer]][src_area][src_pop+layer_map[src_layer]] = syn_pop

    print("Building Model of %u neurons and %u synapses" % (total_neurons, total_synapses))
    build_start_time = perf_counter()
    model.build()
    build_end_time = perf_counter()
    
    duration=args.duration
    duration_timesteps = int(round(duration / DT_MS))
    ten_percent_timestep = duration_timesteps // 10
    print("Loading Model")
    if args.buffer:
        model.load(num_recording_timesteps=args.buffer_size)
    else:
        model.load(num_recording_timesteps=duration_timesteps)

    print("Simulating")

    # Loop through timesteps
    sim_start_time = perf_counter()

    spike_data = {
        area: {pop: [] for pop in neuron_populations[area].keys()}
        for area in neuron_populations.keys()
    }
    flag=0
    while model.t < duration:
        model.step_time()
        if args.buffer:
            if not model.timestep % args.buffer_size:
                model.pull_recording_buffers_from_device()
                record_spike(neuron_populations, spike_data)
        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print("%u%%" % (flag * 10))

    sim_end_time = perf_counter()

    '''
    Saving Spike
    '''
    if not args.buffer:
        model.pull_recording_buffers_from_device()
        record_spike(neuron_populations, spike_data)
    # Merge data
    if args.save_spike:
        save_spike(spike_data)

    visualize(suffix, spike_data, duration=args.duration, model_name=model_name, drop=0, neurons_per_group=200, 
                group_spacing=20, NeuronNumber=NeuronNumber, vis_content=vis_content)


    print("Timing:")
    print("\tBuild:%f" % ((build_end_time - build_start_time) * 1000.0))
    print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

