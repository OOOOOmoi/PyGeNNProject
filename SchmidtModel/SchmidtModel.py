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
from config import collection_params
from visual import visualize
from record import record_spike, save_spike, record_inSyn, save_inSyn
import pynvml, csv
DT_MS=0.1
NUM_THREADS_PER_SPIKE=1
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

def has_key_path(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return False
    return True


def prepare():
    DataPath=os.path.join(current_dir, "default_Data_Model_.json")
    with open(DataPath, 'r') as f:
        ParamOfAll = json.load(f)
    SynapsesNumber=ParamOfAll["synapses"]
    NeuronNumber=ParamOfAll["neuron_numbers"]
    Dist=ParamOfAll["distances"]
    area_list=ParamOfAll["area_list"]
    pop_list=ParamOfAll["population_list"]
    SynapsesWeightMean=ParamOfAll["synapse_weights_mean"]
    SynapsesWeightSd=ParamOfAll["synapse_weights_sd"]
    structure=ParamOfAll["structure"]
    def _convert_vals_to_int(obj):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, (dict, list)):
                    _convert_vals_to_int(v)
                elif isinstance(v, bool) or v is None:
                    continue
                else:
                    try:
                        obj[k] = int(float(v))
                    except Exception:
                        pass
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    _convert_vals_to_int(v)
                elif isinstance(v, bool) or v is None:
                    continue
                else:
                    try:
                        obj[i] = int(float(v))
                    except Exception:
                        pass

    _convert_vals_to_int(NeuronNumber)
    _convert_vals_to_int(SynapsesNumber)
    return NeuronNumber, SynapsesNumber, SynapsesWeightMean, SynapsesWeightSd, Dist, area_list, pop_list, structure

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--duration", type=float, default=1000.0, nargs="?", help="Duration to simulate (ms)")
    parser.add_argument("--buffer", action="store_true", help="Whether use buffer store spike")
    parser.add_argument("--buffer-size", type=int, default=100, nargs="?", help="Size of recording buffer")
    parser.add_argument("--SPARSE", action="store_true", help="Whether use sparse connectivity")
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
    NeuronNumber, SynapsesNumber, SynapsesWeightMean, SynapsesWeightSd, Dist, all_area, pop_list, struct = prepare()
    # area_list = all_area[int(args.AreaIdx)]
    area_list = all_area[0:int(args.AreaNum)]
    if isinstance(area_list, str):
        area_list = [area_list]
    
    rand_str = ''
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    os.makedirs("GenCODE/", exist_ok=True)
    model = GeNNModel("float", "GenCODE/SchmditModel_" + rand_str, device_select_method=DeviceSelect.MANUAL, manual_device_id=args.device)
    model.dt = 0.1
    model.fuse_postsynaptic_models = True
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    
    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})

    neuronParam=collection_params['single_neuron_dict']
    params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                "TauRefrac": neuronParam['t_ref']}
    # print("Creating neuron populations:")
    total_neurons = 0
    neuron_group = 0
    synapse_group = 0
    neuron_populations = defaultdict(dict)
    poisson_init = {"current": 0.0}
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": 0.0}
    rate_ext = collection_params['connection_params']['rate_ext']
    for area in area_list:
        for pop in pop_list:
            if has_key_path(NeuronNumber, area, pop):
                popName = area+pop
                pop_size = NeuronNumber[area][pop]
                if pop_size > 0:
                    neuron_group += 1
                    neuron_pop = model.add_neuron_population(popName, pop_size, "LIF", params, lif_init)

                    if args.poisson:
                        ext_weight = SynapsesWeightMean[area][pop]['external']['external']
                        K = SynapsesNumber[area][pop]['external']['external'] / pop_size
                        rate = rate_ext * K * 1
                        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                        model.add_current_source(area + pop + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)
                    # Enable spike recording
                    neuron_pop.spike_recording_enabled = True

                    # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
                    total_neurons += pop_size
                    neuron_populations[area][pop] = neuron_pop

    total_synapses = 0
    synapse_populations = nested_dict()
    for areaTar, areaSrc in product(area_list, area_list):
        for popTar, popSrc in product(pop_list, pop_list):
            if has_key_path(SynapsesNumber, areaTar, popTar, areaSrc, popSrc):
                wAve = SynapsesWeightMean[areaTar][popTar][areaSrc][popSrc] / 1
                wSd = SynapsesWeightSd[areaTar][popTar][areaSrc][popSrc] / 1
                synNum = SynapsesNumber[areaTar][popTar][areaSrc][popSrc]
                tarName = areaTar+popTar
                srcName = areaSrc+popSrc
                synName = srcName+"2"+tarName
                if areaSrc == areaTar:
                    if 'E' in popSrc:
                        meanDelay=1.5
                        delay_sd=0.75
                        max_d=15.0
                    else:
                        meanDelay=0.75
                        delay_sd=0.375
                        max_d=7.5
                else:
                    meanDelay = Dist[areaSrc][areaTar] / collection_params['connection_params']['interarea_speed']
                    delay_sd = meanDelay * 0.5
                    max_d = meanDelay * 10
                if(synNum>0):
                    synapse_group += 1
                    connect_params = {"num": synNum}
                    d_dist = {"mean": meanDelay, "sd": delay_sd, "min": model.dt, "max": max_d}
                    total_synapses += synNum
                    matrix_type = "SPARSE" if args.SPARSE else "PROCEDURAL"
                    if popSrc.startswith("E"):
                        wAve = abs(wAve)
                        w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                    else:
                        wAve = -abs(wAve)
                        w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                    
                    static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                        {"g": init_var("NormalClipped", w_dist),
                                                        "d": init_var("NormalClippedDelay", d_dist)})
                    syn_pop = model.add_synapse_population(synName, matrix_type,
                            neuron_populations[areaSrc][popSrc], neuron_populations[areaTar][popTar],
                            static_synapse_init, exp_curr_init,
                            init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                    # Set max dendritic delay and span type
                    syn_pop.max_dendritic_delay_timesteps = int(round(max_d / DT_MS))
                    if matrix_type=="PROCEDURAL":
                        syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
                    synapse_populations[areaTar][popTar][areaSrc][popSrc] = syn_pop
                else:
                    synapse_populations[areaTar][popTar][areaSrc][popSrc] = None
    print("Total neurons=%u, total neuron groups=%u, total synapses=%u, total synapse groups=%u" % (total_neurons, neuron_group, total_synapses, synapse_group))

    print("Building Model")
    build_start_time = perf_counter()
    model.build()
    build_end_time = perf_counter()
    print("\tBuild:%f" % ((build_end_time - build_start_time) * 1000.0))
    
    duration=args.duration
    duration_timesteps = int(round(duration / DT_MS))
    ten_percent_timestep = duration_timesteps // 10
    print("Loading Model")
    ld_start_time = perf_counter()
    if args.buffer:
        model.load(num_recording_timesteps=args.buffer_size)
    else:
        model.load(num_recording_timesteps=duration_timesteps)
    ld_end_time = perf_counter()
    print("\tLoad:%f" % ((ld_end_time - ld_start_time) * 1000.0))



    # Loop through timesteps
    sim_start_time = perf_counter()

    spike_data = {
        area: {pop: [] for pop in neuron_populations[area].keys()}
        for area in neuron_populations.keys()
    }
    flag=0
    out_post_history = nested_dict()

    
    pynvml.nvmlInit()

    gpu_index = args.device  # 指定 GPU 编号（如 GPU 0）
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

    for p in processes:
        mem_usage = p.usedGpuMemory / 1024**2
    pynvml.nvmlShutdown()

    # V=[]
    print("Simulating")
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
    for area, area_dict in spike_data.items():
        spike_data_temp = {}
        spike_data_temp[area] = area_dict
        # visualize(suffix, spike_data, duration=args.duration, model_name=model_name, drop=0, neurons_per_group=200, 
        #         group_spacing=20, NeuronNumber=NeuronNumber, vis_content=vis_content)
        visualize(suffix="test", spike_data=spike_data_temp, duration=1000,
                model_name="HMAM", NeuronNumber=NeuronNumber)

    print("Timing:")
    print("\tBuild:%f" % ((build_end_time - build_start_time)))
    print("\tSimulation:%f" % ((sim_end_time - sim_start_time)))

    print("\tInit:%f" % (model.init_time))
    print("\tSparse init:%f" % (model.init_sparse_time))
    print("\tNeuron simulation:%f" % (model.neuron_update_time))
    print("\tSynapse simulation:%f" % (model.presynaptic_update_time))

    filename = 'simulation_results.csv'
    header = ['neuron_groups', 'neurons', 'synapse_groups', 'synapses',
              'build_time_s', 'load_time_s', 'mem_usage', 'sim_time_s']
    
    row = [neuron_group, total_neurons, synapse_group, total_synapses,
           (build_end_time - build_start_time),
           (ld_end_time - ld_start_time),
           mem_usage,
           (sim_end_time - sim_start_time)]
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # 只在第一次写表头
        writer.writerow(row)        # 追加一行数据