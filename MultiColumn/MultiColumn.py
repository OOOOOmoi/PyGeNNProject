import numpy as np
import pygenn.cuda_backend
from scipy.stats import norm
from argparse import ArgumentParser
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from scipy.stats import norm
from time import perf_counter
from itertools import product
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict,defaultdict
from config import AreaList
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)

SynapsesWeightMean=OrderedDict()
SynapsesWeightSd=OrderedDict()
SynapsesNumber=OrderedDict()
NeuronNumber=OrderedDict()
Dist=OrderedDict()
PopList=ParamOfAll['population_list']
SynapsesWeightMean=ParamOfAll["synapse_weights_mean"]
SynapsesWeightSd=ParamOfAll["synapse_weights_sd"]
SynapsesNumber=ParamOfAll["synapses"]
NeuronNumber=ParamOfAll["neuron_numbers"]
Dist=ParamOfAll["distances"]
input = {
    "H1": 501.0 + 10.0,
    "V23": 501.0 + 20,
    "S23": 501.0,
    "E23": 501.0 + 50.0,
    "P23": 501.0 + 50.0,
    "V4": 501.0 + 20.0,
    "S4": 501.0,
    "E4": 501.0 + 50.0,
    "P4": 501.0 + 50.0,
    "V5": 501.0 + 10.0,
    "S5": 501.0 + 10.0,
    "E5": 501.0 + 50.0,
    "P5": 501.0 + 10.0,
    "V6": 501.0 - 0.0,
    "S6": 501.0,
    "E6": 501.0 + 50.0,
    "P6": 501.0 + 10.0
}
TYPE_NAMES = ["E", "I"]
# Simulation timestep [ms]
DT_MS = 0.1

# Means and standard deviations of delays from given source populations (ms)
MEAN_DELAY = {"E": 1.5, "I": 0.75}
DELAY_SD = {"E": 0.75, "I": 0.375}
V = 3.5#mm/s
DELAY_REL = 0.5
max_intra_area_delay = 0
quantile = 0.9999
normal_quantile_cdf = norm.ppf(quantile)
max_delay = {type: MEAN_DELAY[type] + (DELAY_SD[type] * normal_quantile_cdf)
                for type in TYPE_NAMES}
MEAN_W = 87.8e-3  # nA
W_SD = 0.1*MEAN_W
G = -4
NUM_THREADS_PER_SPIKE = 8


def nested_dict():
    return defaultdict(nested_dict)

def getDelayMap():
    delayMap=nested_dict()
    for areaTar, areaSrc in product(AreaList,AreaList):
        for popTar, popSrc in product(PopList, PopList):
            if areaTar == areaSrc:
                if popSrc.startswith("E"):
                    meanDelay = MEAN_DELAY["E"]
                    delay_sd = DELAY_SD["E"]
                    max_d = max_delay["E"]
                else:
                    meanDelay = MEAN_DELAY["I"]
                    delay_sd = DELAY_SD["I"]
                    max_d = max_delay["I"]
            else:
                meanDelay = Dist[areaTar][areaSrc]/V
                delay_sd = meanDelay*DELAY_REL
                max_d = max(max_intra_area_delay, meanDelay + (delay_sd * normal_quantile_cdf))
            delayMap[areaTar][popTar][areaSrc][popSrc]={'ave':meanDelay,'sd':delay_sd,'max':max_d}
    return delayMap

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--duration", type=float, default=1000.0, nargs="?", help="Duration to simulate (ms)")
    parser.add_argument("--stim", type=float, default=0.0, nargs="?", help="Stimulus current to apply to neurons E4 (nA)")
    parser.add_argument("--stimStart", type=float, default=500.0, nargs="?", help="Start time of stimulus (ms)")
    parser.add_argument("--stimEnd", type=float, default=800.0, nargs="?", help="End time of stimulus (ms)")
    parser.add_argument("--neuron-scale", type=float, default=1.0, nargs="?", help="Scaling factor to apply to number of neurons")
    parser.add_argument("--connectivity-scale", type=float, default=1.0, nargs="?", help="Scaling factor to apply to number of neurons")
    parser.add_argument("--buffer", action="store_true", help="Whether use buffer store spike")
    parser.add_argument("--buffer-size", type=int, default=100, nargs="?", help="Size of recording buffer")
    parser.add_argument("--cutff", action="store_true", help="Whether cut ff connections")
    parser.add_argument("--cutfb", action="store_true", help="Whether cut fb connections")
    parser.add_argument("--J", type=float, default=1.0, nargs="?", help="Scaling factor of CC synaptic weights")
    parser.add_argument("--K", type=int, default=1, nargs="?", help="Scale synaptic numbers")
    parser.add_argument("--SPARSE", action="store_true", help="Whether use sparse connectivity")
    return parser

def getModelName(args):
    model_name = "StimExc"
    for area in AreaList:
        model_name += f"_{area}"
    model_name += f"_{args.duration/1000.0:.1f}s"
    if args.stim != 0:
        model_name += f"_stim{args.stim:.1f}nA"
        model_name += f"_start{args.stimStart/1000:.1f}s"
        model_name += f"_end{args.stimEnd/1000:.1f}s"
    if args.neuron_scale != 1.0:
        model_name += f"_Nscale{args.neuron_scale:.1f}"
    if args.connectivity_scale != 1.0:
        model_name += f"_Sscale{args.connectivity_scale:.1f}"
    if args.buffer:
        model_name += f"_buffer{args.buffer_size}"
    if args.cutff:
        model_name += "_cutff"
    if args.cutfb:
        model_name += "_cutfb"
    if args.J != 1.0:
        model_name += f"_J{args.J:.1f}"
    if args.K != 1:
        model_name += f"_K{args.K}"
    return model_name


if __name__ == "__main__":
    args = get_parser().parse_args()
    model_name = getModelName(args)
    with open("output/last_model_name.txt", "w") as f:
        f.write(model_name)
    model = GeNNModel("float", "GenCODE/" + model_name)
    model.dt = DT_MS
    model.fuse_postsynaptic_models = True
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": 2.0}
    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})

    trigger_pulse_model = pygenn.create_current_source_model(
        "trigger_pulse",
        params=["input_time","output_time","magnitude"],  # 参数：噪声强度
        injection_code=
        """
        if (t >= input_time && t < output_time) {
            injectCurrent(magnitude);
        }
        """
    )


    # print("Creating neuron populations:")
    total_neurons = 0
    neuron_populations = defaultdict(dict)
    for area in AreaList:
        for pop in PopList:
            popName = area+pop
            lif_params = {"C": 0.5, "TauM": 20.0, "Vrest": -70.0, "Vreset": -60.0, "Vthresh" : -50.0,
                        "Ioffset": input[pop]/1000.0, "TauRefrac": 2.0}

            pop_size = NeuronNumber[area][pop]*args.neuron_scale
            neuron_pop = model.add_neuron_population(popName, pop_size, "LIF", lif_params, lif_init)
            if AreaList.index(area) == 0 and pop.startswith("E") and args.stim!=0:
                model.add_current_source(pop + '_pulse',
                    trigger_pulse_model, neuron_pop,
                    {   "input_time":args.stimStart,
                        "output_time":args.stimEnd,
                        "magnitude": args.stim/1000.0},
            )

            # Enable spike recording
            neuron_pop.spike_recording_enabled = True

            # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
            total_neurons += pop_size
            neuron_populations[area][pop] = neuron_pop

    total_synapses = 0
    delayMap=getDelayMap()
    for areaTar, areaSrc in product(AreaList,AreaList):
        for popTar, popSrc in product(PopList, PopList):
            wAve = SynapsesWeightMean[areaTar][popTar][areaSrc][popSrc]/1000.0
            wSd = SynapsesWeightSd[areaTar][popTar][areaSrc][popSrc]/1000.0
            if AreaList.index(areaSrc)<AreaList.index(areaTar) and args.cutff:
                continue
            if AreaList.index(areaSrc)>AreaList.index(areaTar) and args.cutfb:
                continue
            synNum = SynapsesNumber[areaTar][popTar][areaSrc][popSrc]*args.connectivity_scale
            tarName = areaTar+popTar
            srcName = areaSrc+popSrc
            synName = srcName+"2"+tarName
            meanDelay=delayMap[areaTar][popTar][areaSrc][popSrc]['ave']
            delay_sd=delayMap[areaTar][popTar][areaSrc][popSrc]['sd']
            max_d=delayMap[areaTar][popTar][areaSrc][popSrc]['max']
            if(synNum>0):
                if areaTar != areaSrc and args.J > 1:
                    wAve = args.J*wAve
                if areaTar != areaSrc and args.K > 1:
                    synNum = int(round(synNum * args.K))
                if areaTar == "V3" and args.J > 1:
                    wAve = wAve/args.J
                if areaSrc == "V3" and args.J > 1:
                    wAve = wAve/args.J
                # print("\tConnection '%s' to '%s': numConnections=%u, meanWeight=%f, weightSD=%f, meanDelay=%f, delaySD=%f"
                # % (srcName, tarName, synNum, wAve, wSd, meanDelay, delay_sd))
                # Build parameters for fixed number total connector
                connect_params = {"num": synNum}
                # Build distribution for delay parameters
                d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_d}
                total_synapses += synNum
                # Build unique synapse name
                matrix_type = "SPARSE" if args.SPARSE else "PROCEDURAL"
                if popSrc.startswith("E"):
                    w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
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
    while model.t < duration:
        model.step_time()
        if args.buffer:
            if not model.timestep % args.buffer_size:
                model.pull_recording_buffers_from_device()
                for area, pop_dict in neuron_populations.items():
                    for pop, p in pop_dict.items():
                        spike_times, spike_ids = p.spike_recording_data[0]
                        spike_data[area][pop].append(np.column_stack((spike_times, spike_ids)))
        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print("%u%%" % (flag * 10))

    sim_end_time = perf_counter()
    if not args.buffer:
        model.pull_recording_buffers_from_device()
        for area, pop_dict in neuron_populations.items():
            for pop, p in pop_dict.items():
                spike_times, spike_ids = p.spike_recording_data[0]
                spike_data[area][pop].append(np.column_stack((spike_times, spike_ids)))
    # Merge data
    for area, pop_dict in spike_data.items():
        for pop, data_chunks in pop_dict.items():
            if len(data_chunks) == 0:
                continue  # 避免没有数据时报错
            all_data = np.vstack(data_chunks)
            np.savetxt(f"output/spike/{area}_{pop}_spikes.csv", all_data, delimiter=",", fmt=("%f", "%d"), header="Times [ms], Neuron ID")


    print("Timing:")
    print("\tBuild:%f" % ((build_end_time - build_start_time) * 1000.0))
    print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

