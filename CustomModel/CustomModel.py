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
from config import collection_params, vis_content, record_I
from getStruct import getWeightMap, getDelayMap, get_struct, has_key_path
from visual import visualize
from connectom import connectom
from record import record_spike, save_spike, record_inSyn, save_inSyn
from expLIF import expLIF_model
DT_MS=0.1
NUM_THREADS_PER_SPIKE=8
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

def prepare(args):
    DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
    with open(DataPath, 'r') as f:
        ParamOfAll = json.load(f)
    SynapsesNumber=ParamOfAll["synapses"]
    NeuronNumber=ParamOfAll["neuron_numbers"]
    Dist=ParamOfAll["distances"]

    model_structure = get_struct()
    SynapsesWeightMean, SynapsesWeightSd = getWeightMap(model_structure, args)
    delayMap = getDelayMap(model_structure, Dist)
    return NeuronNumber, SynapsesNumber, SynapsesWeightMean, SynapsesWeightSd, delayMap

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

def getModelName(args):
    model_name = f"{args.duration/1000.0:.1f}s"
    model_content = collection_params['model_content']
    struct= get_struct()
    for area in struct.keys():
        model_name += f"_{area}"
        if len(struct[area]) != 17:
            for layer in model_content[area]:
                model_name += f"_{layer}"
    if args.stim:
        model_name += f"_stim"
        model_name += f"_start{args.stim_start/1000:.1f}s"
        model_name += f"_end{args.stim_end/1000:.1f}s"
    if args.buffer:
        model_name += f"_buffer{args.buffer_size/1000:.1f}s"
    if args.SPARSE:
        model_name += f"_SPARSE"
    return model_name


if __name__ == "__main__":
    args = parse_all_args()
    model_name = getModelName(args)
    with open("output/last_model_name.txt", "w") as f:
        f.write(model_name)
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    # rand_str = ''
    model = GeNNModel("float", "GenCODE/" + model_name + "_" + rand_str, device_select_method=DeviceSelect.MANUAL, manual_device_id=args.device)
    model.dt = 0.1
    model.fuse_postsynaptic_models = not args.inSyn
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    
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
    NeuronNumber, SynapsesNumber, SynapsesWeightMean, SynapsesWeightSd, delayMap = prepare(args)
    
    struct=get_struct()
    connectom(SynapsesNumber, SynapsesWeightMean, NeuronNumber, struct)
    if "expLIF" in args:
        neuronParam = collection_params['expLIF_dict']
        params = {
            "C": neuronParam['C_m'] / 1000, "TauM": neuronParam['tau_m'],
            "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
            "Vthresh": neuronParam['V_th'], "Ioffset": 0, "TauRefrac": neuronParam['t_ref'],
            "DeltaT": neuronParam['DeltaT'], "VT": neuronParam['VT']
        }
    else:
        neuronParam=collection_params['single_neuron_dict']
        params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                    "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                    "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                    "TauRefrac": neuronParam['t_ref']}
    input=collection_params['connection_params']['input']
    stim_info=collection_params['stim']
    # print("Creating neuron populations:")
    total_neurons = 0
    neuron_populations = defaultdict(dict)
    poisson_init = {"current": 0.0}
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": params['TauRefrac']}
    for area, PopList in struct.items():
        for pop in PopList:
            popName = area+pop
            if pop == "S4" and ("free_scale_input" in args):
                input[pop] += float(args.free_scale_input)
            # if pop == "V4" and ("free_scale" in args):
            #     input[pop] += 10*float(args.free_scale)
            if args.poisson:
                params["Ioffset"] = 0.0
            else:
                params["Ioffset"] = input[pop] / 1000.0
            pop_size = NeuronNumber[area][pop]
            if "expLIF" in args:
                neuron_pop = model.add_neuron_population(popName, pop_size, expLIF_model, params, lif_init)
            else:
                neuron_pop = model.add_neuron_population(popName, pop_size, "LIF", params, lif_init)
            if args.stim and has_key_path(stim_info, area, pop):
                s=stim_info[area][pop]
                model.add_current_source(pop + '_pulse',
                    trigger_pulse_model, neuron_pop,
                    {   "start_time":args.stim_start,
                        "end_time":args.stim_end,
                        "magnitude": s[0]/1000.0},
            )

            if args.poisson:
                ext_weight = SynapsesWeightMean[area][pop]['external']['external'] * 100
                rate = 10 * SynapsesNumber[area][pop]['external']['external'] / NeuronNumber[area][pop]
                poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                model.add_current_source(pop + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)
            # Enable spike recording
            neuron_pop.spike_recording_enabled = True

            # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
            total_neurons += pop_size
            neuron_populations[area][pop] = neuron_pop

    if "CUT" not in args:
        total_synapses = 0
        synapse_populations = nested_dict()
        for areaTar, tarList in struct.items():
            for areaSrc, srcList in struct.items():
                for popTar, popSrc in product(tarList,srcList):
                    wAve = SynapsesWeightMean[areaTar][popTar][areaSrc][popSrc]/1000.0
                    wSd = SynapsesWeightSd[areaTar][popTar][areaSrc][popSrc]/1000.0
                    synNum = SynapsesNumber[areaTar][popTar][areaSrc][popSrc]
                    tarName = areaTar+popTar
                    srcName = areaSrc+popSrc
                    synName = srcName+"2"+tarName
                    meanDelay=delayMap[areaTar][popTar][areaSrc][popSrc]['ave']
                    delay_sd=delayMap[areaTar][popTar][areaSrc][popSrc]['sd']
                    max_d=delayMap[areaTar][popTar][areaSrc][popSrc]['max']
                    if(synNum>0):
                        connect_params = {"num": synNum}
                        # Build distribution for delay parameters
                        d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_d}
                        total_synapses += synNum
                        # Build unique synapse name
                        matrix_type = "SPARSE" if args.SPARSE else "PROCEDURAL"
                        if popSrc.startswith("E"):
                            w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                        else:
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
        print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))

    print("Building Model")
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
    out_post_history = nested_dict()
    # V=[]
    while model.t < duration:
        model.step_time()
        # pop=neuron_populations["V1"]["S4"]
        # pop.vars["V"].pull_from_device()
        # v=pop.vars["V"].current_values
        # V.append(v[0])
        if args.buffer:
            if not model.timestep % args.buffer_size:
                model.pull_recording_buffers_from_device()
                record_spike(neuron_populations, spike_data)
        if args.inSyn:
            record_inSyn(out_post_history, record_I, synapse_populations, PopList)
        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print("%u%%" % (flag * 10))

    # plt.figure(figsize=(8, 5))
    # plt.plot(V)
    # plt.savefig("single_neuron_v.png")

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

    visualize(spike_data, duration=args.duration, model_name=model_name, drop=0, neurons_per_group=200, 
                group_spacing=20, NeuronNumber=NeuronNumber, vis_content=vis_content)

    if args.inSyn:
        save_inSyn(out_post_history)

    print("Timing:")
    print("\tBuild:%f" % ((build_end_time - build_start_time) * 1000.0))
    print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

