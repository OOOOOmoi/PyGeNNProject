import numpy as np
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
from config import LayerList, Area, TYPE_NAMES, getWeightMap, plot_effective_weight_heatmap, connection_params
from nested_dict import nested_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
with open(DataPath,'r') as f:
    ParamOfAll=json.load(f)
SynapsesNumber=OrderedDict()
NeuronNumber=OrderedDict()
SynapsesWeightMean=OrderedDict()
SynapsesWeightSd=OrderedDict()
PopList=ParamOfAll['population_list']
SynapsesNumber=ParamOfAll["synapses"]
NeuronNumber=ParamOfAll["neuron_numbers"]
# SynapsesWeightMean=ParamOfAll["synapse_weights_mean"]
# SynapsesWeightSd=ParamOfAll["synapse_weights_sd"]

input = connection_params['input']

DataPath=os.path.join(current_dir, "Wsolution.json")
with open(DataPath,'r') as f:
    Wsolution=json.load(f)

# Simulation timestep [ms]
DT_MS = 0.1

# Means and standard deviations of delays from given source populations (ms)
MEAN_DELAY = {"E": 1.5, "I": 0.75}
DELAY_SD = {"E": 0.75, "I": 0.375}
V = 3.5#mm/s
DELAY_REL = 0.5
NUM_THREADS_PER_SPIKE = 8
max_intra_area_delay = 0
quantile = 0.9999
normal_quantile_cdf = norm.ppf(quantile)
max_delay = {type_: MEAN_DELAY[type_] + (DELAY_SD[type_] * normal_quantile_cdf)
                for type_ in ["E", "I"]}

def nested_dict():
    return defaultdict(nested_dict)

def getDelayMap():
    delayMap=nested_dict()
    for layerTar, layerSrc in product(LayerList,LayerList):
        for typeTar, typeSrc in product(TYPE_NAMES, TYPE_NAMES):
            if layerSrc == "1":
                typeSrc = "H"
            if layerTar == "1":
                typeTar = "H"
            srcName = typeSrc + layerSrc
            tarName = typeTar + layerTar
            if typeSrc=="E":
                meanDelay = MEAN_DELAY["E"]
                delay_sd = DELAY_SD["E"]
                max_d = max_delay["E"]
            else:
                meanDelay = MEAN_DELAY["I"]
                delay_sd = DELAY_SD["I"]
                max_d = max_delay["I"]
            delayMap[tarName][srcName]={'ave':meanDelay,'sd':delay_sd,'max':max_d}
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
    parser.add_argument("--inSyn", action="store_true", help="Whether record inSyn")
    return parser

def getModelName(args):
    model_name = Area
    for l in LayerList:
        model_name += f"_{l}"
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

    total_neurons = 0
    neuron_populations = defaultdict(dict)
    for l in LayerList:
        for type_ in TYPE_NAMES:
            if l == "1":
                type_ = "H"
                pop = type_+l
                lif_params = {"C": 0.5, "TauM": 20.0, "Vrest": -70.0, "Vreset": -60.0, "Vthresh" : -50.0,
                            "Ioffset": input[pop]/1000.0, "TauRefrac": 2.0}

                pop_size = NeuronNumber[Area][pop]
                neuron_pop = model.add_neuron_population(pop, pop_size, "LIF", lif_params, lif_init)

                # Enable spike recording
                neuron_pop.spike_recording_enabled = True

                # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
                total_neurons += pop_size
                neuron_populations[pop] = neuron_pop
                break
            pop = type_+l
            lif_params = {"C": 0.5, "TauM": 20.0, "Vrest": -70.0, "Vreset": -60.0, "Vthresh" : -50.0,
                        "Ioffset": input[pop]/1000.0, "TauRefrac": 2.0}

            pop_size = NeuronNumber[Area][pop]
            neuron_pop = model.add_neuron_population(pop, pop_size, "LIF", lif_params, lif_init)

            # Enable spike recording
            neuron_pop.spike_recording_enabled = True

            # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
            total_neurons += pop_size
            neuron_populations[pop] = neuron_pop

    total_synapses = 0
    delayMap=getDelayMap()
    synapse_populations = nested_dict()
    for layerTar, layerSrc in product(LayerList,LayerList):
        for typeTar, typeSrc in product(TYPE_NAMES, TYPE_NAMES):
            if layerSrc == "1":
                typeSrc = "H"
            if layerTar == "1":
                typeTar = "H"
            srcName = typeSrc + layerSrc
            tarName = typeTar + layerTar
            SynapsesWeightMean, SynapsesWeightSd = getWeightMap()
            wAve = SynapsesWeightMean[tarName][srcName]/1000.0
            wSd = SynapsesWeightSd[tarName][srcName]/1000.0
            # wAve = Wsolution[srcName] * 1e07
            # wSd = Wsolution[srcName] * 0.1
            synNum = SynapsesNumber[Area][tarName][Area][srcName]
            synName = srcName+"2"+tarName
            meanDelay=delayMap[tarName][srcName]['ave']
            delay_sd=delayMap[tarName][srcName]['sd']
            max_d=delayMap[tarName][srcName]['max']
            if(synNum>0):
                # Build parameters for fixed number total connector
                connect_params = {"num": synNum}
                # Build distribution for delay parameters
                d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_d}
                total_synapses += synNum
                # Build unique synapse name
                matrix_type = "SPARSE"
                if typeSrc=="E":
                    w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                    static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                        {"g": init_var("NormalClipped", w_dist),
                                                        "d": init_var("NormalClippedDelay", d_dist)})
                    syn_pop = model.add_synapse_population(synName, matrix_type,
                    neuron_populations[srcName], neuron_populations[tarName],
                    static_synapse_init, exp_curr_init,
                    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                    # Set max dendritic delay and span type
                    syn_pop.max_dendritic_delay_timesteps = int(round(max_d / DT_MS))
                    syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
                else:
                    w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                    static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                        {"g": init_var("NormalClipped", w_dist),
                                                        "d": init_var("NormalClippedDelay", d_dist)})
                    syn_pop = model.add_synapse_population(synName, matrix_type,
                    neuron_populations[srcName], neuron_populations[tarName],
                    static_synapse_init, exp_curr_init,
                    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                    # Set max dendritic delay and span type
                    syn_pop.max_dendritic_delay_timesteps = int(round(max_d / DT_MS))
                    syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
                synapse_populations[tarName][srcName] = syn_pop
            else:
                synapse_populations[tarName][srcName] = None

    print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))

    plot_effective_weight_heatmap(
        SynapsesWeightMean,
        SynapsesNumber, 
        NeuronNumber, 
        areaName=Area,
        title='Normalized W × In-degree Heatmap'
    )

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
        pop: [] for pop in neuron_populations.keys()
    }
    flag=0
    out_post_history = nested_dict()
    while model.t < duration:
        model.step_time()
        if args.buffer:
            if not model.timestep % args.buffer_size:
                model.pull_recording_buffers_from_device()
                for pop, p in neuron_populations.items():
                    spike_times, spike_ids = p.spike_recording_data[0]
                    spike_data[pop].append(np.column_stack((spike_times, spike_ids)))

        
        if args.inSyn:
            tar_pop = "S4"
            for src_pop in PopList:
                if synapse_populations["V1"][tar_pop]["V1"][src_pop] is not None:
                    synapse_populations["V1"][tar_pop]["V1"][src_pop].out_post.pull_from_device()
                    out_post_array = synapse_populations["V1"][tar_pop]["V1"][src_pop].out_post.view[:,:20]
                    if isinstance(out_post_history["V1"][tar_pop]["V1"][src_pop], dict):
                        out_post_history["V1"][tar_pop]["V1"][src_pop] = []
                    out_post_history["V1"][tar_pop]["V1"][src_pop].append(out_post_array.copy())

        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print("%u%%" % (flag * 10))

        

    sim_end_time = perf_counter()

    '''
    Saving Spike
    '''
    if not args.buffer:
        model.pull_recording_buffers_from_device()
        for pop, p in neuron_populations.items():
                spike_times, spike_ids = p.spike_recording_data[0]
                spike_data[pop].append(np.column_stack((spike_times, spike_ids)))
    # Merge data
    for pop, data_chunks in spike_data.items():
        if len(data_chunks) == 0:
            continue  # 避免没有数据时报错
        all_data = np.vstack(data_chunks)
        os.makedirs(f"output/spike/{Area}/", exist_ok=True)
        np.savetxt(f"output/spike/{Area}/{Area}_{pop}_spikes.csv", all_data, delimiter=",", fmt=("%f", "%d"), header="Times [ms], Neuron ID")


    print("Timing:")
    print("\tBuild:%f" % ((build_end_time - build_start_time) * 1000.0))
    print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

    