"""
冒烟测试: 验证 PyGeNN 5.2 中若干非标准用法的可行性
1) 自定义稀疏连接片段 (row_build_code, 无放回抽样, 排除自连接)
2) 自定义 HH 型神经元模型 (无重置, 用 spiked 状态位实现单次尖峰)
3) 权重更新模型: pre_neuron_var_refs 在 synapse_dynamics_code 中读取突触前膜电位
4) 自定义突触后模型: inSyn 作为电导 + var_refs 读取突触后膜电位
5) out_post 每步 pull / 尖峰记录
"""
import numpy as np
import pygenn
from pygenn import (GeNNModel, VarAccess, init_postsynaptic, init_sparse_connectivity,
                    init_weight_update, init_var, create_var_ref)
from pygenn.cuda_backend import DeviceSelect

N = 5

# ---------- 1. 神经元 ----------
wb = pygenn.create_neuron_model(
    "smoke_WB_HH",
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

# ---------- 2. 突触后模型: inSyn 作为电导 ----------
psm = pygenn.create_postsynaptic_model(
    "smoke_GABAA_cond",
    params=[("E_syn", "scalar")],
    vars=[("G_syn", "scalar", VarAccess.READ_WRITE)],
    neuron_var_refs=[("V", "scalar")],
    sim_code="""
        G_syn = inSyn;
        injectCurrent(G_syn * (V - E_syn));
    """,
)

# ---------- 3. 权重更新: 分级释放 + s 一阶动力学 ----------
wu = pygenn.create_weight_update_model(
    "smoke_GABAA_graded",
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

# ---------- 4. 稀疏连接: 宿主机侧生成, 与 MATLAB 完全一致的拓扑 ----------
# 每个突触后神经元无放回抽取 num 个突触前来源, 排除自连接
rng = np.random.default_rng(1996)
M = 4
pre_list, post_list = [], []
for j in range(N):
    pool = np.delete(np.arange(N), j)
    src = rng.choice(pool, size=min(M, N - 1), replace=False)
    pre_list.append(src)
    post_list.append(np.full(len(src), j))
pre_inds = np.concatenate(pre_list)
post_inds = np.concatenate(post_list)

model = GeNNModel("float", "smoke_wb1996", device_select_method=DeviceSelect.MANUAL,
                  manual_device_id=0)
model.dt = 0.02
model.timing_enabled = True

nparams = {"gL": 0.1, "EL": -65.0, "gNa": 35.0, "ENa": 55.0, "gK": 9.0, "EK": -90.0,
           "phi": 5.0, "C": 1.0, "Iapp": 1.0, "Vthresh": -20.0}
ninit = {"V": init_var("Uniform", {"min": -70.0, "max": -50.0}),
         "h": 0.6, "n": 0.3, "spiked": 0.0}
pop = model.add_neuron_population("inh", N, wb, nparams, ninit)
pop.spike_recording_enabled = True
pop.vars["V"].recording_enabled = True

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
n_steps = int(round(200.0 / model.dt))
model.load(num_recording_timesteps=n_steps)

field = []
while model.t < 200.0 - 1e-9:
    model.step_time()
    syn.out_post.pull_from_device()
    field.append(float(np.mean(syn.out_post.view[:])))

model.pull_recording_buffers_from_device()
spk_t, spk_id = pop.spike_recording_data[0]
Vrec = pop.vars["V"].values
print("SMOKE OK")
print("spikes:", len(spk_t), "field mean:", float(np.mean(field)), "field max:", float(np.max(field)))
print("V shape:", Vrec.shape, "V range: %.2f .. %.2f" % (Vrec.min(), Vrec.max()))
