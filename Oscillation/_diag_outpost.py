# 诊断: out_post 的 shape/重置行为, V 的正确读取方式
import numpy as np
import pygenn
from pygenn import (GeNNModel, VarAccess, init_postsynaptic, init_sparse_connectivity,
                    init_weight_update, init_var, create_var_ref)
from pygenn.cuda_backend import DeviceSelect

N = 5
wb = pygenn.create_neuron_model(
    "diag_WB_HH",
    params=["gL", "EL", "gNa", "ENa", "gK", "EK", "phi", "C", "Iapp", "Vthresh"],
    vars=[("V", "scalar", VarAccess.READ_WRITE),
          ("h", "scalar", VarAccess.READ_WRITE),
          ("n", "scalar", VarAccess.READ_WRITE),
          ("spiked", "scalar", VarAccess.READ_WRITE)],
    sim_code="""
        const scalar u_m = -0.1 * (V + 35.0);
        scalar a_m;
        if (u_m > -1.0e-6 && u_m < 1.0e-6) { a_m = 1.0; }
        else { a_m = u_m / (exp(u_m) - 1.0); }
        const scalar b_m = 4.0 * exp(-(V + 60.0) / 18.0);
        const scalar m_inf = a_m / (a_m + b_m);
        const scalar a_h = 0.07 * exp(-(V + 58.0) / 20.0);
        const scalar b_h = 1.0 / (exp(-0.1 * (V + 28.0)) + 1.0);
        const scalar u_n = -0.1 * (V + 34.0);
        scalar a_n;
        if (u_n > -1.0e-6 && u_n < 1.0e-6) { a_n = 0.1; }
        else { a_n = 0.1 * u_n / (exp(u_n) - 1.0); }
        const scalar b_n = 0.125 * exp(-(V + 44.0) / 80.0);
        const scalar I_Na = gNa * m_inf * m_inf * m_inf * h * (V - ENa);
        const scalar I_K  = gK * n * n * n * n * (V - EK);
        const scalar I_L  = gL * (V - EL);
        const scalar dV   = (-I_Na - I_K - I_L - Isyn + Iapp) / C;
        V += dV * dt;
        h += phi * (a_h * (1.0 - h) - b_h * h) * dt;
        n += phi * (a_n * (1.0 - n) - b_n * n) * dt;
        h = fmin(1.0, fmax(0.0, h));
        n = fmin(1.0, fmax(0.0, n));
        if (V < Vthresh - 5.0) { spiked = 0.0; }
    """,
    threshold_condition_code="(V >= Vthresh) && (spiked == 0.0)",
    reset_code="spiked = 1.0;",
)
psm = pygenn.create_postsynaptic_model(
    "diag_GABAA_cond",
    params=[("E_syn", "scalar")],
    vars=[("G_syn", "scalar", VarAccess.READ_WRITE)],
    neuron_var_refs=[("V", "scalar")],
    sim_code="""
        G_syn = inSyn;
        injectCurrent(G_syn * (V - E_syn));
        inSyn = 0.0;
    """,
)
wu = pygenn.create_weight_update_model(
    "diag_GABAA_graded",
    params=[("alpha_syn", "scalar"), ("beta_syn", "scalar"), ("theta_syn", "scalar")],
    vars=[("w", "scalar", VarAccess.READ_WRITE),
          ("s", "scalar", VarAccess.READ_WRITE)],
    pre_neuron_var_refs=[("V_pre", "scalar")],
    synapse_dynamics_code="""
        const scalar F = 1.0 / (1.0 + exp(-(V_pre - theta_syn) / 2.0));
        s += (alpha_syn * F * (1.0 - s) - beta_syn * s) * dt;
        addToPost(w * s);
    """,
)
rng = np.random.default_rng(1996)
pre_list, post_list = [], []
for j in range(N):
    pool = np.delete(np.arange(N), j)
    src = rng.choice(pool, size=4, replace=False)
    pre_list.append(src); post_list.append(np.full(4, j))
pre_inds = np.concatenate(pre_list); post_inds = np.concatenate(post_list)

model = GeNNModel("float", "diag_wb1996", device_select_method=DeviceSelect.MANUAL,
                  manual_device_id=0)
model.dt = 0.02
nparams = {"gL": 0.1, "EL": -65.0, "gNa": 35.0, "ENa": 55.0, "gK": 9.0, "EK": -90.0,
           "phi": 5.0, "C": 1.0, "Iapp": 1.0, "Vthresh": -20.0}
ninit = {"V": init_var("Uniform", {"min": -70.0, "max": -50.0}),
         "h": 0.6, "n": 0.3, "spiked": 0.0}
pop = model.add_neuron_population("inh", N, wb, nparams, ninit)
pop.spike_recording_enabled = True
syn = model.add_synapse_population(
    "inh_inh", "SPARSE", pop, pop,
    init_weight_update(wu, {"alpha_syn": 2.0, "beta_syn": 0.1, "theta_syn": 0.0},
                       {"w": 0.1 / 4.0, "s": 0.0},
                       pre_var_refs={"V_pre": create_var_ref(pop, "V")}),
    init_postsynaptic(psm, {"E_syn": -75.0}, {"G_syn": 0.0},
                      var_refs={"V": create_var_ref(pop, "V")}),
    init_sparse_connectivity("Uninitialised", {}),
)
model.build()
syn.set_sparse_connections(pre_inds, post_inds)
model.load(num_recording_timesteps=10000)

print("out_post.view.shape =", syn.out_post.view.shape)
print("out_post dtype =", syn.out_post.view.dtype)

for i in range(200):
    model.step_time()
    if i in (0, 1, 2, 50, 100, 199):
        syn.out_post.pull_from_device()
        pop.vars["V"].pull_from_device()
        pop.vars["spiked"].pull_from_device()
        print("t=%6.2f  out_post=%s  V=%s" % (
            model.t, np.array2string(syn.out_post.view, precision=4, max_line_width=200),
            np.array2string(pop.vars["V"].view, precision=2)))

model.pull_recording_buffers_from_device()
spk_t, spk_id = pop.spike_recording_data[0]
print("spikes:", len(spk_t))
print("spike ids:", np.unique(spk_id, return_counts=True))
