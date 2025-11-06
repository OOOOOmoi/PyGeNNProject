import numpy as np
from argparse import ArgumentParser, Namespace
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic, create_var_init_snippet,
                    init_sparse_connectivity, init_weight_update, init_var, 
                    create_weight_update_model)
from pygenn.cuda_backend import DeviceSelect
import matplotlib.pyplot as plt

single_neuron_dict = {
    'E_L': -70.0, # mV
    'V_th': -50.0, # mV
    'V_reset': -60.0, # mV
    'C_m': 500.0, # pF
    'tau_m': 20.0, # ms
    'tau_syn': 0.5, # ms
    't_ref': 2.0 # ms
}

constant_init = create_var_init_snippet(
    "constant_init",
    params=[("constant", "scalar")],
    var_init_code=
    """
    value = constant;
    """
)

dual_exp = create_weight_update_model(
    "dual_exp",
    params=[("taur", "scalar"), ("taud", "scalar"), ("weight", "scalar"), ("DT", "scalar")],
    vars=[("x", "scalar", pygenn.VarAccess.READ_WRITE), 
          ("g", "scalar", pygenn.VarAccess.READ_WRITE)],

    synapse_dynamics_code=
    """
    addToPost(weight * g);
    scalar dt = t - st_pre;
    scalar inp = 0;
    if (dt <= 0.9 * DT) {
        inp = 1 / DT;
    }
    scalar dx = (-x / taur + inp) * DT;
    x += dx;
    scalar dg = (-g / taud + x) * DT;
    g += dg;
    """
)

constant_current = create_weight_update_model(
    "constant_current",
    params=[("weight", "scalar")],
    pre_spike_syn_code="addToPost(weight);"
)

constant_syn = pygenn.create_postsynaptic_model(
    "constant_syn",
    sim_code=
    """
    injectCurrent(inSyn);
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

model_name = "DualEXP"
model = GeNNModel("float", model_name)
model.dt = 0.1
model.fuse_postsynaptic_models = True
model.default_narrow_sparse_ind_enabled = True
model.timing_enabled = True
model.default_var_location = VarLocation.HOST_DEVICE
model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
params = {"C": single_neuron_dict['C_m']/1000, "TauM": single_neuron_dict['tau_m'],
            "Vrest": single_neuron_dict['E_L'], "Vreset": single_neuron_dict['V_reset'],
            "Vthresh" : single_neuron_dict['V_th'], "Ioffset": 0.501,
            "TauRefrac": single_neuron_dict['t_ref']}
lif_init = {"V": -70.0, "RefracTime": 0.0}
dual_exp_params = {"taur": 5.0, "taud": 100.0, "weight": 0.1, "DT": model.dt}
dual_exp_var_init = {"x": 0.0, "g": 0.0}
constant_current_params = {"weight": 0.1}
connect_params = {"num": 1}
pre = model.add_neuron_population("pre", 1, "LIF", params, lif_init)
params["Ioffset"] = 0.0
post = model.add_neuron_population("post", 1, "LIF", params, lif_init)
static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                {"g": 0.1,
                                "d": 1.5})
# dual_exp_pop = model.add_synapse_population(
#     "syn1", "SPARSE", pre, post, 
#     init_weight_update(dual_exp, dual_exp_params, dual_exp_var_init),
#     init_postsynaptic("DeltaCurr"),
#     init_sparse_connectivity("OneToOne")
# )
dual_exp_pop = model.add_synapse_population(
    "syn2", "SPARSE", pre, post,
    init_weight_update(constant_current, constant_current_params),
    init_postsynaptic(postsyn_dual_exp, {"taur": 5.0, "taud": 100.0}, {"g": 0.0}),
    init_sparse_connectivity("OneToOne")
)
dual_exp_pop.max_dendritic_delay_timesteps = 15
duration = 1000.0
model.build()
model.load(num_recording_timesteps = int(round(duration / model.dt)))
out_post_history=[]
V_pre=[]
V_post=[]
while model.t < duration:
    model.step_time()
    dual_exp_pop.out_post.pull_from_device()
    if model.t == 200.0:
        dual_exp_pop.out_post.view[:] += 0.1
    dual_exp_pop.out_post.push_to_device()
    out_post_array = dual_exp_pop.out_post.view[:]
    out_post_history.append(out_post_array.copy())

    pre.vars["V"].pull_from_device()
    v=pre.vars["V"].current_values
    V_pre.append(v[0])

    post.vars["V"].pull_from_device()
    v=post.vars["V"].current_values
    V_post.append(v[0])

all_data = np.vstack(out_post_history)  # 合并所有时间片

np.savetxt("dual_exp.csv", all_data, delimiter=",", fmt="%.5f")

fig, ax = plt.subplots(3, figsize=(30, 10), sharex=True)
ax[0].set_xlabel("Time step (0.1 ms)")
ax[0].set_ylabel("Membrane potential (mV)")
ax[0].plot(V_pre, color='blue')  # 用 ax.plot 而不是 plt.plot

ax[1].set_xlabel("Time step (0.1 ms)")
ax[1].set_ylabel("Membrane potential (mV)")
ax[1].plot(V_post, color='orange')  # 用 ax.plot 而不是 plt.plot

ax[2].set_xlabel("Time step (0.1 ms)")
ax[2].set_ylabel("Post-synaptic current")
ax[2].plot(all_data[:,0], color='green')  # 用 ax.plot 而不是 plt.plot

fig.tight_layout()
fig.savefig("dual_exp.png")