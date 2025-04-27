import numpy as np

from argparse import ArgumentParser
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from scipy.stats import norm
from time import perf_counter
from itertools import product
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_synapse_number = os.path.join(current_dir, "SynapsesNumber.txt")
file_weight = os.path.join(current_dir, "SynapsesWeight.txt")

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Layer names
popName = ["H1","E23","S23","P23","V23",
            "E4","S4","P4","V4",
            "E5","S5","P5","V5",
            "E6","S6","P6","V6"]
popNum = {
        "H1": 15545,
        "E23": 51738,
        "S23": 1892,
        "P23": 2610,
        "V23": 4514,
        "E4": 74933,
        "S4": 4041,
        "P4": 7037,
        "V4": 1973,
        "E5": 21624,
        "S5": 1586,
        "P5": 1751,
        "V5": 334,
        "E6": 20278,
        "S6": 1667,
        "P6": 1656,
        "V6": 302
    }
input = {
    "H1": 501.0,
    "V23": 501.0 - 10.0,
    "S23": 501.0,
    "E23": 501.0 + 50.0,
    "P23": 501.0,
    "V4": 501.0 - 50.0,
    "S4": 501.0,
    "E4": 501.0 + 50.0,
    "P4": 501.0 + 10.0,
    "V5": 501.0 - 10.0,
    "S5": 501.0,
    "E5": 501.0 + 10.0,
    "P5": 501.0,
    "V6": 501.0 - 10.0,
    "S6": 501.0,
    "E6": 501.0 + 50.0,
    "P6": 501.0
}

def get_syn_num():
    syn_number = {}
    try:
        with open(file_synapse_number, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) != 3:
                    continue 
                src_pop, tar_pop, syn_num = parts[0], parts[1], int(float(parts[2]))
                
                if tar_pop not in syn_number:
                    syn_number[tar_pop] = {}
                syn_number[tar_pop][src_pop] = syn_num
    except FileNotFoundError:
        print(f"Error: File '{file_synapse_number}' not found.")
    
    return syn_number

def get_syn_weight():
    syn_weight = {}
    try:
        with open(file_weight, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) != 4:
                    continue  
                src_pop, tar_pop, w_ave, w_sd = parts[0], parts[1], float(parts[2]) / 1000, float(parts[3]) / 1000
                
                if tar_pop not in syn_weight:
                    syn_weight[tar_pop] = {}
                syn_weight[tar_pop][src_pop] = {"wAve": w_ave, "wSd": w_sd}
    except FileNotFoundError:
        print(f"Error: File '{file_weight}' not found.")
    return syn_weight

# Population Type
TYPE_NAMES = ["E", "I"]

# Simulation timestep [ms]
DT_MS = 0.1

# Means and standard deviations of delays from given source populations (ms)
MEAN_DELAY = {"E": 1.5, "I": 0.75}
DELAY_SD = {"E": 0.75, "I": 0.375}

MEAN_W = 87.8e-3  # nA
W_SD = 0.1*MEAN_W
G = -4
NUM_THREADS_PER_SPIKE = 8


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--duration", type=float, default=1000.0, help="Duration to simulate (ms)")
    parser.add_argument("--neuron-scale", type=float, default=1.0, help="Scaling factor to apply to number of neurons")
    parser.add_argument("--connectivity-scale", type=float, default=1.0, help="Scaling factor to apply to number of neurons")
    parser.add_argument("--kernel-profiling", action="store_true", help="Output kernel profiling data")
    parser.add_argument("--save-data", action="store_true", help="Output kernel profiling data")
    parser.add_argument("--buffer-size", type=int, default=100, help="Size of recording buffer")
    return parser

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_parser().parse_args()
    # ----------------------------------------------------------------------------
    # Network creation
    # ----------------------------------------------------------------------------
    model = GeNNModel("float", "Singlecolumn")
    model.dt = DT_MS
    model.fuse_postsynaptic_models = True
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = args.kernel_profiling
    model.default_var_location = VarLocation.DEVICE
    model.default_sparse_connectivity_location = VarLocation.DEVICE

    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": 2.0}
    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})

    quantile = 0.9999
    normal_quantile_cdf = norm.ppf(quantile)
    max_delay = {type: MEAN_DELAY[type] + (DELAY_SD[type] * normal_quantile_cdf)
                for type in TYPE_NAMES}
    print("Max excitatory delay:%fms , max inhibitory delay:%fms" % (max_delay["E"], max_delay["I"]))

    # Calculate maximum dendritic delay slots
    # **NOTE** it seems inefficient using maximum for all but this allows more aggressive merging of postsynaptic models
    max_dendritic_delay_slots = int(round(max(max_delay.values()) / DT_MS))
    print("Max dendritic delay slots:%d" % max_dendritic_delay_slots)

    print("Creating neuron populations:")
    total_neurons = 0
    neuron_populations = {}
    for pop in popName:

        lif_params = {"C": 0.5, "TauM": 20.0, "Vrest": -70.0, "Vreset": -60.0, "Vthresh" : -50.0,
                    "Ioffset": input[pop]/1000.0, "TauRefrac": 2.0}

        pop_size = popNum[pop]*args.neuron_scale
        neuron_pop = model.add_neuron_population(pop, pop_size, "LIF", lif_params, lif_init)

        # Enable spike recording
        neuron_pop.spike_recording_enabled = True

        print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (pop, pop_size, input[pop]/1000.0))
        total_neurons += pop_size
        neuron_populations[pop] = neuron_pop

    print("Creating synapse populations:")
    synWeight=get_syn_weight()
    synNum=get_syn_num()
    total_synapses = 0
    for tar_pop, src_pop in product(popName, popName):

        # Determine mean weight
        # Specialised weights
        mean_weight = synWeight[tar_pop][src_pop]['wAve']
        weight_sd = synWeight[tar_pop][src_pop]['wSd']
        G = 1

        # Identical presynaptic neuron types have identical weight distributions
        # mean_weight=MEAN_W
        # weight_sd = W_SD

        # Weight set to 0
        # mean_weight = 0
        # weight_sd = 0

        # Calculate number of connections
        num_connections = synNum[tar_pop][src_pop]*args.connectivity_scale
        if src_pop.startswith("E"):
            meanDelay= MEAN_DELAY["E"]
            delay_sd= DELAY_SD["E"]
            max_d= max_delay["E"]
        else:
            meanDelay= MEAN_DELAY["I"]
            delay_sd= DELAY_SD["I"]
            max_d= max_delay["I"]
        if num_connections > 0:
            print("\tConnection '%s' to '%s': numConnections=%u, meanWeight=%f, weightSD=%f, meanDelay=%f, delaySD=%f"
                % (src_pop, tar_pop, num_connections, mean_weight, weight_sd, meanDelay, delay_sd))

            # Build parameters for fixed number total connector
            connect_params = {"num": num_connections}
            # Build distribution for delay parameters
            d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_d}
            total_synapses += num_connections
            # Build unique synapse name
            synapse_name = tar_pop + "2" + src_pop
            matrix_type = "SPARSE"

            # Excitatory
            if src_pop.startswith("E"):
                # Build distribution for weight parameters
                # **HACK** np.float32 doesn't seem to automatically cast
                w_dist = {"mean": mean_weight, "sd": weight_sd, "min": 0.0, "max": float(np.finfo(np.float32).max)}

                # Create weight parameters
                static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                        {"g": init_var("NormalClipped", w_dist),
                                                        "d": init_var("NormalClippedDelay", d_dist)})
                # Add synapse population
                syn_pop = model.add_synapse_population(synapse_name, matrix_type,
                    neuron_populations[src_pop], neuron_populations[tar_pop],
                    static_synapse_init, exp_curr_init,
                    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                # Set max dendritic delay and span type
                syn_pop.max_dendritic_delay_timesteps = max_dendritic_delay_slots
                if matrix_type=="PROCEDURAL":
                    syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
            # Inhibitory
            else:
                # Build distribution for weight parameters
                # **HACK** np.float32 doesn't seem to automatically cast
                w_dist = {"mean": G*mean_weight, "sd": weight_sd, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                # Create weight parameters
                static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                        {"g": init_var("NormalClipped", w_dist),
                                                        "d": init_var("NormalClippedDelay", d_dist)})
                # Add synapse population
                syn_pop = model.add_synapse_population(synapse_name, matrix_type,
                    neuron_populations[src_pop], neuron_populations[tar_pop],
                    static_synapse_init, exp_curr_init,
                    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

                # Set max dendritic delay and span type
                syn_pop.max_dendritic_delay_timesteps = max_dendritic_delay_slots
                if matrix_type=="PROCEDURAL":
                    syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE

    print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))

    print("Building Model")
    model.build()

    duration=args.duration
    duration_timesteps = int(round(duration / DT_MS))
    ten_percent_timestep = duration_timesteps // 10
    print("Loading Model")
    model.load(num_recording_timesteps=args.buffer_size)

    print("Simulating")

    # Loop through timesteps
    sim_start_time = perf_counter()

    spike_data = {n: [] for n in neuron_populations.keys()}

    while model.t < duration:
        # Advance simulation
        # print(model.timestep)
        model.step_time()

        # Indicate every 10%
        if not model.timestep % args.buffer_size:
            model.pull_recording_buffers_from_device()
            for n, pop in neuron_populations.items():
                spike_times, spike_ids = pop.spike_recording_data[0]
                spike_data[n].append(np.column_stack((spike_times, spike_ids)))

        if (model.timestep % ten_percent_timestep) == 0:
            print("%u%%" % (model.timestep / 100))

    sim_end_time = perf_counter()

    # Merge data
    if args.save_data:
        for n, data_chunks in spike_data.items():
            all_data = np.vstack(data_chunks)
            np.savetxt(f"output/{n}_spikes.csv", all_data, delimiter=",", fmt=("%f", "%d"), header="Times [ms], Neuron ID")


    print("Timing:")
    print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

    # # Create plot
    figure, axes = plt.subplots(2, 1)

    # **YUCK** re-order neuron populations for plotting
    ordered_neuron_populations = list(reversed(list(neuron_populations.values())))

    start_id = 0
    bar_y = 0.0
    for pop in ordered_neuron_populations:
        all_spike_data = np.vstack(spike_data[pop.name])
        spike_times, spike_ids = all_spike_data[:, 0], all_spike_data[:, 1]

        actor = axes[0].scatter(spike_times, spike_ids + start_id, s=2, edgecolors="none")

        axes[1].barh(bar_y, len(spike_times) / (float(pop.num_neurons) * duration / 1000.0),
                    align="center", color=actor.get_facecolor(), ecolor="black")

        start_id += pop.num_neurons

        bar_y += 1.0

    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Neuron number")

    axes[1].set_xlabel("Mean firing rate [Hz]")
    axes[1].set_yticks(np.arange(0.0, len(neuron_populations), 1.0))
    axes[1].set_yticklabels([n.name for n in ordered_neuron_populations])

    plt.savefig("raster_plot.jpg")