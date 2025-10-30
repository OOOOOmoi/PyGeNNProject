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


single_neuron_dict = {
    'rate_ext':{
        "H1": 650,
        "E23": 930, "S23": 870.0,  "P23": 1460.0,  "V23": 1405.0,
        "E4": 890,  "S4": 2105.0,   "P4": 1984.0,   "V4": 240.0,
        "E5": 4740,  "S5": 530.0,   "P5": 930.0,  "V5": 870.0,
        "E6": 1770,  "S6": 885.0,    "P6": 1170.0,    "V6": 1620.0,
    },

    'Cm': {# pF
        "H1": 37.11,
        "E23": 123.41, "P23": 70.95, "S23": 82.34, "V23": 41.23,
        "E4": 80.16, "P4": 81.21, "S4": 132.86, "V4": 40.3,
        "E5": 149.43, "P5": 70.9, "S5": 52.32, "V5": 59.29,
        "E6": 99.96, "P6": 49.65, "S6": 96.09, "V6": 65.87,
    },
    'gL': {# nS
        "H1": 4.07,
        "E23": 2.47, "P23": 9.49, "S23": 3.17, "V23": 6.4,
        "E4": 5.16, "P4": 9.19, "S4": 7.96, "V4": 1.87,
        "E5": 16.66, "P5": 5.21, "S5": 3.43, "V5": 6.52,
        "E6": 5.88, "P6": 6.86, "S6": 2.99, "V6": 6.09,
    },
    'tref': {# ms
        "H1": 3.5,
        "E23": 3.0, "P23": 1.26, "S23": 1.85, "V23": 2.75,
        "E4": 4.4, "P4": 1.5, "S4": 2.2, "V4": 2.4,
        "E5": 4.25, "P5": 1.85, "S5": 1.9, "V5": 2.55,
        "E6": 3.3, "P6": 1.65, "S6": 2.1, "V6": 2.85,
    },
    'Vrest': {# mV
        "H1": -65.5,
        "E23": -80.97, "P23": -82.35, "S23": -69.16, "V23": -67.94,
        "E4": -72.53, "P4": -70.45, "S4": -74.2, "V4": -63.14,
        "E5": -68.28, "P5": -77.5, "S5": -70.01, "V5": -72.00,
        "E6": -77.5, "P6": -76.42, "S6": -62.99, "V6": -78.85,
    },
    'Vth': {# mV
        "H1": -40.20,
        "E23": -40.53, "P23": -56.32, "S23": -39.95, "V23": -41.34,
        "E4": -47.63, "P4": -44.23, "S4": -44.07, "V4": -40.89,
        "E5": -40.55, "P5": -51.2, "S5": -47.38, "V5": -51.2,
        "E6": -42.31, "P6": -49.06, "S6": -37.19, "V6": -44.81,
    },
}

model = GeNNModel("float", "ScalingTestModel")
model.dt = 0.1
model.default_narrow_sparse_ind_enabled = True
model.timing_enabled = True
model.default_var_location = VarLocation.HOST_DEVICE
model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})
srcType_ = 'E23'
tarType_ = 'P23'
Cm = single_neuron_dict['Cm']
gL = single_neuron_dict['gL']
tref = single_neuron_dict['tref']
Vrest = single_neuron_dict['Vrest']
Vth = single_neuron_dict['Vth']
rate_ext = single_neuron_dict['rate_ext']

params = {"C": 0, "TauM": 0, "Vrest": 0, "Vreset": 0, "Vthresh" : 0, "Ioffset": 0, "TauRefrac": 0}
srcParam = params
srcParam["C"] = Cm[srcType_] / 1000
srcParam["TauM"] = Cm[srcType_] / gL[srcType_]
srcParam["Vrest"] = Vrest[srcType_]
srcParam["Vreset"] = Vrest[srcType_] - 10
srcParam["Vthresh"] = Vth[srcType_]
srcParam["TauRefrac"] = tref[srcType_]
srcParam["Ioffset"] = 0.15

lif_init = {"V": srcParam["Vreset"], "RefracTime": srcParam['TauRefrac']}
src_pop = model.add_neuron_population('src', 1, "LIF", params, lif_init)

tarParam = params
tarParam["C"] = Cm[tarType_] / 1000
tarParam["TauM"] = Cm[tarType_] / gL[tarType_]
tarParam["Vrest"] = Vrest[tarType_]
tarParam["Vreset"] = Vrest[tarType_] - 10
tarParam["Vthresh"] = Vth[tarType_]
tarParam["TauRefrac"] = tref[tarType_]

lif_init = {"V": tarParam["Vreset"], "RefracTime": tarParam['TauRefrac']}
tar_pop = model.add_neuron_population('tar', 1, "LIF", tarParam, lif_init)

tarParam["TauM"] += 10
tar_pop2 = model.add_neuron_population('tar2', 1, "LIF", tarParam, lif_init)

tarParam["TauM"] += 20
tar_pop3 = model.add_neuron_population('tar3', 1, "LIF", tarParam, lif_init)


C_m = tarParam["C"]
tau_m = tarParam["TauM"]
tau_syn = 2.0
weight = ((C_m**(-1) * tau_m * tau_syn / (tau_syn - tau_m) *
            ((tau_m / tau_syn) ** (- tau_m / (tau_m - tau_syn)) -
            (tau_m / tau_syn) ** (- tau_syn / (tau_m - tau_syn)))) ** (-1))
wEE_init = {"g": weight / 1000}
exc_post_syn_params = {"tau": tau_syn}
connect_params = {"num": 1}
syn = model.add_synapse_population("syn", "SPARSE",
    src_pop, tar_pop,
    init_weight_update("StaticPulseConstantWeight", wEE_init),
    init_postsynaptic("ExpCurr", exc_post_syn_params),
    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

tau_m += 10
weight = ((C_m**(-1) * tau_m * tau_syn / (tau_syn - tau_m) *
            ((tau_m / tau_syn) ** (- tau_m / (tau_m - tau_syn)) -
            (tau_m / tau_syn) ** (- tau_syn / (tau_m - tau_syn)))) ** (-1))
wEE_init = {"g": weight / 1000}
syn2 = model.add_synapse_population("syn2", "SPARSE",
    src_pop, tar_pop2,
    init_weight_update("StaticPulseConstantWeight", wEE_init),
    init_postsynaptic("ExpCurr", exc_post_syn_params),
    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))

tau_m += 10
weight = ((C_m**(-1) * tau_m * tau_syn / (tau_syn - tau_m) *
            ((tau_m / tau_syn) ** (- tau_m / (tau_m - tau_syn)) -
            (tau_m / tau_syn) ** (- tau_syn / (tau_m - tau_syn)))) ** (-1))
wEE_init = {"g": weight / 1000}
syn3 = model.add_synapse_population("syn3", "SPARSE",
    src_pop, tar_pop3,
    init_weight_update("StaticPulseConstantWeight", wEE_init),
    init_postsynaptic("ExpCurr", exc_post_syn_params),
    init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))


model.build()
duration = 1000.0
duration_steps = int(duration / model.dt)

model.load(num_recording_timesteps=duration_steps)
# out_post_history=[]
V_pre=[]
V_post=[]
V_post2=[]
V_post3=[]
while model.t < duration:
    model.step_time()
    # syn.out_post.pull_from_device()
    # out_post_array = syn.out_post.view[:,:]
    # out_post_history.append(out_post_array.copy())

    src_pop.vars["V"].pull_from_device()
    v=src_pop.vars["V"].current_values
    V_pre.append(v[0])

    tar_pop.vars["V"].pull_from_device()
    v=tar_pop.vars["V"].current_values
    V_post.append(v[0])

    tar_pop2.vars["V"].pull_from_device()
    v=tar_pop2.vars["V"].current_values
    V_post2.append(v[0])

    tar_pop3.vars["V"].pull_from_device()
    v=tar_pop3.vars["V"].current_values
    V_post3.append(v[0])

# all_data = np.vstack(out_post_history)  # 合并所有时间片

# np.savetxt("dual_exp.csv", all_data, delimiter=",", fmt="%.5f")

fig, ax = plt.subplots(2, figsize=(20, 10), sharex=True)
ax[0].set_xlabel("Time step (0.1 ms)")
ax[0].set_ylabel("Membrane potential (mV)")
ax[0].plot(V_pre, color='blue')  # 用 ax.plot 而不是 plt.plot

ax[1].set_xlabel("Time step (0.1 ms)")
ax[1].set_ylabel("Membrane potential (mV)")
ax[1].plot(V_post)  # 用 ax.plot 而不是 plt.plot
ax[1].plot(V_post2)
ax[1].plot(V_post3)
# ax[2].set_xlabel("Time step (0.1 ms)")
# ax[2].set_ylabel("Post-synaptic current")
# ax[2].plot(all_data[:,0], color='green')  # 用 ax.plot 而不是 plt.plot

fig.tight_layout()
fig.savefig("scalingTest.png")
