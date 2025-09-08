import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update
import pygenn
import os
from argparse import ArgumentParser, Namespace
import datetime
import random
import string
from pygenn.cuda_backend import DeviceSelect
import sys
current_dir = os.path.dirname(__file__)

def generate_unique_suffix(length=3):
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"

def smooth_firing_rate(spike_times, total_neurons, sample_bin=1, sigma=3, drop=200):
    t_min = drop
    t_max = int(spike_times.max()) + 1
    time_bins = np.arange(t_min, t_max + 1, sample_bin)
    binned_rate, _ = np.histogram(spike_times, bins=time_bins)
    binned_rate = binned_rate * 1000 / total_neurons
    return gaussian_filter1d(binned_rate.astype(float), sigma=sigma), time_bins

def record_inSyn(out_post_history, synapse_populations):
    syn_pop=synapse_populations
    syn_pop.out_post.pull_from_device()
    out_post_array = syn_pop.out_post.view[:,:20]
    out_post_history.append(out_post_array.copy())

def save_inSyn(out_post_history, name):
    data = out_post_history

    if not data:
        return -1  # 空数据跳过

    all_data = np.vstack(data)  # 合并所有时间片

    filename = f"{name}.csv"
    fileDir = os.path.join(current_dir, filename)
    np.savetxt(fileDir, all_data, delimiter=",", fmt="%.3f")

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--wEE", type=float, default=0, nargs="?", help="weight of EE connections")
    parser.add_argument("--wEI", type=float, default=0, nargs="?", help="weight of EI connections")
    parser.add_argument("--wIE", type=float, default=0, nargs="?", help="weight of IE connections")
    parser.add_argument("--wII", type=float, default=0, nargs="?", help="weight of II connections")
    parser.add_argument("--device", type=int, default=0, help="Device ID to use for simulation")
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

expLIF_model = pygenn.create_neuron_model(
    "expLIF",
    params=["Vthresh", "TauM", "TauRefrac", "C", "Vrest", "Vreset", "Ioffset", "DeltaT", "VT"],
    vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE),
          ("RefracTime", "scalar", pygenn.VarAccess.READ_WRITE)],
    sim_code=
        """
        if (RefracTime <= 0.0) {
            scalar dV = (-(V - Vrest) + DeltaT * exp((V - VT) / DeltaT) + Rmembrane * (Ioffset + Isyn)) / TauM;
            V += dV * dt;
        }else {
            RefracTime -= dt;
        }
        """,
    threshold_condition_code="(RefracTime <= 0.0) && (V >= Vthresh)",
    reset_code=
        """
        V = Vreset;
        RefracTime = TauRefrac;
        """,
    derived_params=[("Rmembrane", lambda pars,dt:pars["TauM"]/pars["C"])],
)

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

postsyn_dual_exp = pygenn.create_postsynaptic_model(
    "dual_exp_post",
    params=[("taur", "scalar"), ("taud", "scalar")],
    vars=[("g", "scalar")],
    sim_code= """
        injectCurrent(g);
        g += (-g/taud + inSyn)*dt;
        inSyn += -inSyn/taur*dt; 
    """
)

args = parse_all_args()
model_name = "wEE_{}_wEI_{}_wIE_{}_wII_{}".format(args.wEE, args.wEI, args.wIE, args.wII)
suffix = generate_unique_suffix()
# suffix = ''
model = GeNNModel("float", "CODE/EIBalance_"+suffix, device_select_method=DeviceSelect.MANUAL, manual_device_id=args.device)
model.dt = 0.1
model.fuse_postsynaptic_models = False

single_neuron_dict = {"C_m": 0.5, "tau_m": 20.0, "Vrest": -60.0, "Vreset": -60.0,
              "V_th": -45.0, "input": 0, "t_ref": 5.0, "DeltaT": 5.0, "VT": -50.0}

explif_params = {
    "C": single_neuron_dict['C_m'],  # Convert pF to nF
    "TauM": single_neuron_dict['tau_m'],
    "Vrest": single_neuron_dict['Vrest'],
    "Vreset": single_neuron_dict['Vreset'],
    "Vthresh": single_neuron_dict['V_th'],
    "Ioffset": single_neuron_dict['input'],
    "TauRefrac": single_neuron_dict['t_ref'],
    "DeltaT": single_neuron_dict['DeltaT'],
    "VT": single_neuron_dict['VT']
}

lif_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
            "RefracTime": 0.0}

exc_pop = model.add_neuron_population("E", 4000, expLIF_model, explif_params, lif_init)
model.add_current_source("E_pulse", trigger_pulse_model, exc_pop,
                         {"start_time":500,
                        "end_time":3000,
                        "magnitude":0.2})
explif_params["Ioffset"]=0
inh_pop = model.add_neuron_population("I", 1000, expLIF_model, explif_params, lif_init)
model.add_current_source("I_pulse", trigger_pulse_model, inh_pop,
                         {"start_time":500,
                        "end_time":3000,
                        "magnitude":0.15})

exc_pop.spike_recording_enabled = True
inh_pop.spike_recording_enabled = True

wEE_init = {"g": args.wEE}
wEI_init = {"g": args.wEI}
wII_init = {"g": -1*args.wII}
wIE_init = {"g": -1*args.wIE}

exc_post_syn_params = {"tau": 2.0}
inh_post_syn_params = {"tau": 5.0}

fixed_prob = {"prob": 0.02}

# EEpop=model.add_synapse_population("EE", "SPARSE",
#     exc_pop, exc_pop,
#     init_weight_update("StaticPulseConstantWeight", wEE_init),
#     init_postsynaptic("ExpCurr", exc_post_syn_params),
#     init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))
EEpop=model.add_synapse_population("EE_NMDA", "SPARSE",
    exc_pop, exc_pop,
    init_weight_update("StaticPulseConstantWeight", wEE_init),
    init_postsynaptic(postsyn_dual_exp, {"taur": 5.0, "taud": 100.0}, {"g": 0.0}),
    init_sparse_connectivity("FixedProbability", fixed_prob))

EIpop=model.add_synapse_population("EI", "SPARSE",
    exc_pop, inh_pop,
    init_weight_update("StaticPulseConstantWeight", wEI_init),
    init_postsynaptic("ExpCurr", exc_post_syn_params),
    init_sparse_connectivity("FixedProbability", fixed_prob))

IIpop=model.add_synapse_population("II", "SPARSE",
    inh_pop, inh_pop,
    init_weight_update("StaticPulseConstantWeight", wII_init),
    init_postsynaptic("ExpCurr", inh_post_syn_params),
    init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

IEpop=model.add_synapse_population("IE", "SPARSE",
    inh_pop, exc_pop,
    init_weight_update("StaticPulseConstantWeight", wIE_init),
    init_postsynaptic("ExpCurr", inh_post_syn_params),
    init_sparse_connectivity("FixedProbability", fixed_prob))

model.build()
model.load(num_recording_timesteps=50000)
# EEi = []
# EIi = []
# IIi = []
# IEi = []
while model.t < 5000:
    model.step_time()
#     record_inSyn(EEi, EEpop)
#     record_inSyn(EIi, EIpop)
#     record_inSyn(IIi, IIpop)
#     record_inSyn(IEi, IEpop)
# save_inSyn(EEi, "EE")
# save_inSyn(EIi, "EI")
# save_inSyn(IIi, "II")
# save_inSyn(IEi, "IE")

model.pull_recording_buffers_from_device()

exc_spike_times, exc_spike_ids = exc_pop.spike_recording_data[0]
inh_spike_times, inh_spike_ids = inh_pop.spike_recording_data[0]

fig, axes = plt.subplots(4, sharex=True, figsize=(20, 10))

# Define some bins to calculate spike rates
bin_size = 5

# Plot excitatory and inhibitory spikes on first axis
axes[0].scatter(exc_spike_times, exc_spike_ids, s=1)
axes[0].scatter(inh_spike_times, inh_spike_ids + 4000, s=1)

# Plot excitatory rates on second axis
exc_rate, exc_bin = smooth_firing_rate(exc_spike_times, 4000, sample_bin=bin_size, drop=200)

# Plot inhibitory rates on third axis
inh_rate, inh_bin = smooth_firing_rate(inh_spike_times, 1000, sample_bin=bin_size, drop=200)
exc_rate_normal = exc_rate / max(exc_rate)
inh_rate_normal = inh_rate / max(inh_rate)
# exc_rate = gaussian_filter1d(exc_rate, sigma=2)  # sigma 单位是 bin
# inh_rate = gaussian_filter1d(inh_rate, sigma=2)
time_axis = (exc_bin[:-1] + exc_bin[1:]) / 2
axes[1].plot(time_axis, exc_rate)
axes[3].plot(time_axis, exc_rate_normal, label='Excitatory', color='blue')
time_axis = (inh_bin[:-1] + inh_bin[1:]) / 2
axes[2].plot(time_axis, inh_rate)
axes[3].plot(time_axis, inh_rate_normal, label='Inhibitory', color='orange')

# Label axes
axes[0].set_ylabel("Neuron ID")
axes[1].set_ylabel("Excitatory rate [Hz]")
axes[2].set_ylabel("Inhibitory rate [Hz]")
axes[3].set_xlabel("Time [ms]")
axes[3].set_ylabel("Normalized rate")
axes[3].legend()
axes[0].set_title(f"Raster plot and firing rates - {model_name}", fontsize=20)
os.makedirs('output', exist_ok=True)
plt.savefig(f"output/{model_name}.png")