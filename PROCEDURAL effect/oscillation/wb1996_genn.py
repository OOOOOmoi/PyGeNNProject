#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wang & Buzsaki (1996) ING (Interneuron Network Gamma) 模型的 PyGeNN 5.2 实现
==================================================================================
论文: "Gamma Oscillation by Synaptic Inhibition in a Hippocampal Interneuronal
      Network Model", J. Neurosci. 16(20):6402-6413

模型要点(与 MATLAB 复现版 wb1996_*.m 严格对齐):
  - N=100 个单室 HH 型中间神经元 (m 用瞬时值 m_inf, phi=5 温度因子)
  - GABAA 抑制性突触, 分级释放: F(V_pre) = 1/(1+exp(-V_pre/2))
    ds/dt = alpha*F(V_pre)*(1-s) - beta*s,  alpha=2.0/ms, beta=0.1/ms (tau=10ms)
  - 论文电导归一化 (p3: "gsyn is divided by Msyn"): 每突触 w = 0.1/Msyn mS/cm^2,
    总抑制电导恒为 0.1 mS/cm^2, 与连接度无关
  - 突触电流 I_syn = sum(w*s) * (V_post - E_syn), E_syn = -75 mV
  - 连接: 每个突触后神经元无放回抽取 Msyn 个突触前来源, 排除自连接

GeNN 实现关键:
  - 神经元无硬重置: 用 spiked 状态位实现"上穿 -20mV 触发一次尖峰"
  - WUM 用 synapse_dynamics_code 每步积分 s 并 addToPost(w*s)
  - PSM sim_code 注入电导电流后手动 inSyn=0 (DeltaCurr 模式),
    否则 inSyn 会跨步累积 (GeNN 对连续突触动力学不自动清零)
  - 连接用 DENSE 矩阵 (N=100 仅 10^4 权重): 本版 PyGeNN 的 SPARSE +
    set_sparse_connections 存在紧凑 ind 与内核填充步长错位的 bug 会越界,
    自定义连接片段转译器又不支持运行时长度数组; w=0 突触无贡献

用法:
  python wb1996_genn.py                     # 默认: Msyn=100 (全连接), T=1000ms
  python wb1996_genn.py --Msyn 60           # 稀疏连接 (论文 Fig.8 部分同步)
  python wb1996_genn.py --Msyn 30 --tag desync
  python wb1996_genn.py --iapp-std 0.05     # Iapp 加异质性
"""
import argparse
import os
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygenn
from pygenn import (GeNNModel, VarAccess, init_postsynaptic, init_sparse_connectivity,
                    init_weight_update, init_var, create_var_ref)
from pygenn.cuda_backend import DeviceSelect

# ----------------------------------------------------------------------------
# 神经元模型: Wang-Buzsaki 单室 HH 中间神经元
#   C dV/dt = Iapp - I_Na - I_K - I_L - I_syn
#   I_Na = gNa*m_inf^3*h*(V-ENa), I_K = gK*n^4*(V-EK), I_L = gL*(V-EL)
#   dh/dt = phi*(a_h*(1-h) - b_h*h), dn/dt 同理
# Iapp 做成 READ_ONLY var, 以便按神经元数组初始化(支持异质性)
# ----------------------------------------------------------------------------
WB_HH = pygenn.create_neuron_model(
    "WB1996_HH",
    params=["gL", "EL", "gNa", "ENa", "gK", "EK", "phi", "C", "Vthresh"],
    vars=[("V", "scalar", VarAccess.READ_WRITE),
          ("h", "scalar", VarAccess.READ_WRITE),
          ("n", "scalar", VarAccess.READ_WRITE),
          ("spiked", "scalar", VarAccess.READ_WRITE),
          ("Iapp", "scalar", VarAccess.READ_ONLY)],
    sim_code="""
        // alpha_m = 0.1*(V+35)/(1-exp(-0.1*(V+35))), 奇异点 V=-35 极限为 1
        const scalar u_m = -0.1 * (V + 35.0);
        scalar a_m;
        if (u_m > -1.0e-6 && u_m < 1.0e-6) { a_m = 1.0; }
        else { a_m = u_m / (exp(u_m) - 1.0); }
        const scalar b_m = 4.0 * exp(-(V + 60.0) / 18.0);
        const scalar m_inf = a_m / (a_m + b_m);

        const scalar a_h = 0.07 * exp(-(V + 58.0) / 20.0);
        const scalar b_h = 1.0 / (exp(-0.1 * (V + 28.0)) + 1.0);

        // alpha_n = 0.01*(V+34)/(1-exp(-0.1*(V+34))), 奇异点极限 0.1
        const scalar u_n = -0.1 * (V + 34.0);
        scalar a_n;
        if (u_n > -1.0e-6 && u_n < 1.0e-6) { a_n = 0.1; }
        else { a_n = 0.1 * u_n / (exp(u_n) - 1.0); }
        const scalar b_n = 0.125 * exp(-(V + 44.0) / 80.0);

        const scalar I_Na = gNa * m_inf * m_inf * m_inf * h * (V - ENa);
        const scalar I_K  = gK * n * n * n * n * (V - EK);
        const scalar I_L  = gL * (V - EL);
        V += ((-I_Na - I_K - I_L - Isyn + Iapp) / C) * dt;

        h += phi * (a_h * (1.0 - h) - b_h * h) * dt;
        n += phi * (a_n * (1.0 - n) - b_n * n) * dt;
        h = fmin(1.0, fmax(0.0, h));
        n = fmin(1.0, fmax(0.0, n));

        // 尖峰结束后解除状态位, 允许下一次上穿触发
        if (V < Vthresh - 5.0) { spiked = 0.0; }
    """,
    threshold_condition_code="(V >= Vthresh) && (spiked == 0.0)",
    reset_code="spiked = 1.0;",
)

# ----------------------------------------------------------------------------
# 突触后模型: inSyn 作为 GABAA 电导, 注入电流后手动清零 (DeltaCurr 模式)
# ----------------------------------------------------------------------------
GABAA_PSM = pygenn.create_postsynaptic_model(
    "WB1996_GABAA_cond",
    params=[("E_syn", "scalar")],
    vars=[("G_syn", "scalar", VarAccess.READ_WRITE)],
    neuron_var_refs=[("V", "scalar")],
    sim_code="""
        G_syn = inSyn;
        injectCurrent(G_syn * (V - E_syn));
        inSyn = 0.0;
    """,
)

# ----------------------------------------------------------------------------
# 权重更新模型: 分级释放 + s 一阶动力学 (论文式 2.4)
#   F(V_pre) = 1/(1+exp(-(V_pre - theta)/2))
#   ds/dt = alpha*F(V_pre)*(1-s) - beta*s
#   每步向突触后送入电导 w*s
# ----------------------------------------------------------------------------
GABAA_GRADED_WU = pygenn.create_weight_update_model(
    "WB1996_GABAA_graded",
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


def build_connectivity(N, Msyn, seed, w_per_syn):
    """每个突触前神经元无放回抽取 Msyn 个突触后目标, 排除自连接 (与 MATLAB
    wb1996_run_network.m 一致: 固定出度, 入度 ~ Poisson(Msyn) 涨落)。
    入度涨落提供结构异质性, 是 M 越小同步越差的关键 —— 若改为固定入度,
    每个神经元接收的总电导完全相同, 网络在任意 M 下都近乎完美同步。

    返回稠密权重矩阵 W (行=突触前, 列=突触后, 行主序展平)。
    注: 使用 DENSE 矩阵而非 SPARSE —— 本版 PyGeNN 的 set_sparse_connections
    生成紧凑 ind, 但 SPARSE 内核按 num_post 填充步长寻址, 二者错位导致越界。
    N=100 时稠密仅 10^4 权重, GPU 开销可忽略; w=0 的突触无贡献。
    """
    rng = np.random.default_rng(seed)
    M = min(Msyn, N - 1)
    W = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        pool = np.delete(np.arange(N), i)
        tgts = rng.choice(pool, size=M, replace=False)
        W[i, tgts] = w_per_syn
    return W


def compute_kappa(spk_t, spk_id, N, t0, t1, bin_ms=5.0):
    """论文相干性度量: 稳态段 spike count (bin_ms 时间窗) 的神经元对 Pearson 相关平均"""
    n_bins = int((t1 - t0) / bin_ms)
    counts = np.zeros((N, n_bins))
    sel = (spk_t >= t0) & (spk_t < t1)
    idx = ((spk_t[sel] - t0) / bin_ms).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    np.add.at(counts, (spk_id[sel].astype(int), idx), 1)
    active = np.where(counts.sum(axis=1) > 0)[0]
    if len(active) < 2:
        return 0.0
    v = counts[active].var(axis=1)
    good = active[v > 0]
    if len(good) < 2:
        return 0.0
    C = np.corrcoef(counts[good])
    ll = np.tril(np.ones((len(good), len(good)), bool), -1)
    kk = float(np.mean(C[ll]))
    return 0.0 if np.isnan(kk) else kk


def main():
    ap = argparse.ArgumentParser(description="Wang-Buzsaki 1996 ING model in PyGeNN")
    ap.add_argument("--Msyn", type=int, default=100, help="每细胞突触输入数 (100=全连接)")
    ap.add_argument("--T", type=float, default=1000.0, help="仿真时长 ms")
    ap.add_argument("--dt", type=float, default=0.02, help="积分步长 ms")
    ap.add_argument("--iapp", type=float, default=1.0, help="平均驱动电流 uA/cm^2")
    ap.add_argument("--iapp-std", type=float, default=0.0, help="Iapp 异质性 (正态 std)")
    ap.add_argument("--seed", type=int, default=1996, help="随机种子")
    ap.add_argument("--gpu", type=int, default=0, help="GPU 设备号 (仅用 0/1)")
    ap.add_argument("--tag", type=str, default="", help="输出文件名后缀")
    args = ap.parse_args()

    # ---- 网络参数 (论文标定) ----
    N = 100
    Msyn = min(args.Msyn, N)
    gsyn_total = 0.1                  # mS/cm^2, 论文归一化: 总电导恒定
    w_per_syn = gsyn_total / Msyn     # 每突触电导 = 0.1/Msyn
    alpha_syn, beta_syn, theta_syn = 2.0, 0.1, 0.0   # ms^-1
    E_syn = -75.0                     # mV
    t_transient = 100.0               # 瞬态剔除
    gamma_lo, gamma_hi = 20.0, 80.0

    tag = args.tag if args.tag else f"M{Msyn}"
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 64)
    print("Wang & Buzsaki (1996) ING - PyGeNN 实现")
    print(f"N={N}, Msyn={Msyn}, w={w_per_syn:.5f} mS/cm^2/突触 (总电导 {gsyn_total})")
    print(f"Iapp={args.iapp} + N(0,{args.iapp_std}), T={args.T} ms, dt={args.dt} ms")
    print("=" * 64)

    rng = np.random.default_rng(args.seed)

    # ---- 连接 (宿主机侧生成稠密权重矩阵) ----
    W = build_connectivity(N, Msyn, args.seed, w_per_syn)
    n_syns = int((W > 0).sum())
    print(f"连接: {n_syns} 个有效突触 ({Msyn}/细胞, 无自连接, 无重复), DENSE 矩阵")

    # ---- 模型 ----
    model = GeNNModel("float", f"wb1996_{tag}",
                      device_select_method=DeviceSelect.MANUAL,
                      manual_device_id=args.gpu)
    model.dt = args.dt
    model.timing_enabled = True

    # 初始条件与 MATLAB 对齐: V ~ clip(N(-65, 2^2), -75, -50), h=0.6, n=0.3, s=0
    V0 = np.clip(-65.0 + 2.0 * rng.standard_normal(N), -75.0, -50.0)
    Iapp_arr = args.iapp + (args.iapp_std * rng.standard_normal(N)
                            if args.iapp_std > 0 else np.zeros(N))

    nparams = {"gL": 0.1, "EL": -65.0, "gNa": 35.0, "ENa": 55.0, "gK": 9.0,
               "EK": -90.0, "phi": 5.0, "C": 1.0, "Vthresh": -20.0}
    ninit = {"V": V0, "h": 0.6, "n": 0.3, "spiked": 0.0, "Iapp": Iapp_arr}
    pop = model.add_neuron_population("inh", N, WB_HH, nparams, ninit)
    pop.spike_recording_enabled = True

    syn = model.add_synapse_population(
        "inh_inh", "DENSE", pop, pop,
        init_weight_update(GABAA_GRADED_WU,
                           {"alpha_syn": alpha_syn, "beta_syn": beta_syn,
                            "theta_syn": theta_syn},
                           {"w": W.flatten(), "s": 0.0},
                           pre_var_refs={"V_pre": create_var_ref(pop, "V")}),
        init_postsynaptic(GABAA_PSM, {"E_syn": E_syn}, {"G_syn": 0.0},
                          var_refs={"V": create_var_ref(pop, "V")}),
    )

    t0 = _time.time()
    model.build()
    n_steps = int(round(args.T / args.dt))
    model.load(num_recording_timesteps=n_steps)
    print(f"构建+加载完成 ({_time.time()-t0:.1f}s), 开始仿真 {n_steps} 步...")

    # ---- 仿真循环: 每 rec_every 步记录一次 V (展示用) ----
    rec_every = 5                     # 0.1 ms
    n_rec = n_steps // rec_every
    V_rec = np.zeros((n_rec, N), dtype=np.float32)
    t_rec = np.arange(1, n_rec + 1) * rec_every * args.dt

    t0 = _time.time()
    ir = 0
    for step in range(1, n_steps + 1):
        model.step_time()
        if step % rec_every == 0 and ir < n_rec:
            pop.vars["V"].pull_from_device()
            V_rec[ir] = pop.vars["V"].current_view
            ir += 1
    sim_wall = _time.time() - t0
    print(f"仿真完成: {sim_wall:.1f}s 墙钟 ({args.T/sim_wall:.0f} ms/s)")

    model.pull_recording_buffers_from_device()
    spk_t, spk_id = pop.spike_recording_data[0]
    spk_t = np.asarray(spk_t, dtype=float)
    spk_id = np.asarray(spk_id, dtype=int)
    print(f"总尖峰数: {len(spk_t)}")

    # ---- 分析 ----
    sel = spk_t >= t_transient
    rates = np.bincount(spk_id[sel], minlength=N) / ((args.T - t_transient) / 1000.0)
    mean_rate, std_rate = float(rates.mean()), float(rates.std())

    # PSTH (1 ms bin) 群体放电率
    bin_ms = 1.0
    edges = np.arange(t_transient, args.T + bin_ms, bin_ms)
    psth, _ = np.histogram(spk_t[sel], bins=edges)
    pop_rate = psth / (bin_ms * 1e-3) / N          # Hz/神经元
    t_psth = edges[:-1] + bin_ms / 2

    # FFT 功率谱 (去趋势, 全段)
    x = pop_rate - pop_rate.mean()
    n_fft = len(x)
    fs = 1000.0 / bin_ms
    P = np.abs(np.fft.rfft(x)) ** 2
    f_ax = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    # 全局峰值 (排除 <5Hz 低频)
    search = f_ax >= 5.0
    pk = np.argmax(P[search])
    pk_idx = np.where(search)[0][0] + pk
    peak_freq = float(f_ax[pk_idx])
    # Gamma 带内峰值 (尖峰串脉冲的 PSTH 频谱含谐波, 全局峰值可能是 2 次谐波)
    gmask = (f_ax >= gamma_lo) & (f_ax <= gamma_hi)
    g_idx_local = np.argmax(P[gmask])
    g_idx = np.where(gmask)[0][0] + g_idx_local
    gamma_peak_freq = float(f_ax[g_idx])
    pmask = f_ax >= 5.0
    gamma_ratio = 100.0 * P[gmask].sum() / max(P[pmask].sum(), 1e-12)

    kappa = compute_kappa(spk_t, spk_id, N, t_transient, args.T, bin_ms=5.0)

    in_gamma = gamma_lo <= peak_freq <= gamma_hi
    print("-" * 64)
    print(f"平均放电率: {mean_rate:.1f} ± {std_rate:.1f} Hz")
    print(f"PSTH 全局峰值频率: {peak_freq:.1f} Hz  "
          f"({'在' if in_gamma else '不在'} Gamma 频段)")
    print(f"PSTH Gamma 带内峰值: {gamma_peak_freq:.1f} Hz")
    print(f"Gamma 频段功率占比: {gamma_ratio:.1f}%")
    print(f"相干性 kappa(5ms): {kappa:.3f}")
    print("-" * 64)

    # ---- 绘图 (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(spk_t[sel], spk_id[sel], '.', ms=1.5, color='k')
    ax.set_xlim(t_transient, args.T)
    ax.set_ylim(0, N + 1)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron #")
    ax.set_title(f"Raster (Msyn={Msyn})")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_psth, pop_rate, color='0.7', lw=0.6, label="1 ms bin")
    # 高斯平滑 (sigma=5 ms)
    sig = 5.0
    kg = np.arange(-15, 16)
    kern = np.exp(-0.5 * (kg / sig) ** 2)
    kern /= kern.sum()
    sm = np.convolve(pop_rate, kern, mode="same")
    ax.plot(t_psth, sm, color='r', lw=1.2, label="Gaussian $\sigma$=5 ms")
    ax.set_xlim(t_transient, args.T)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Pop. rate (Hz/neuron)")
    ax.set_title("Population firing rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    Pn = 10 * np.log10(P / max(P.max(), 1e-12) + 1e-12)
    ax.plot(f_ax, Pn, color='b', lw=1.0)
    ax.axvspan(gamma_lo, gamma_hi, color='r', alpha=0.12)
    ax.axvline(gamma_lo, color='r', ls='--', lw=0.8)
    ax.axvline(gamma_hi, color='r', ls='--', lw=0.8)
    ax.plot(peak_freq, Pn[pk_idx], 'ro')
    ax.annotate(f"global {peak_freq:.1f} Hz", (peak_freq, Pn[pk_idx]),
                textcoords="offset points", xytext=(8, -2), color='r',
                fontweight='bold')
    ax.plot(gamma_peak_freq, Pn[g_idx], 'rs', mfc='none')
    ax.annotate(f"gamma {gamma_peak_freq:.1f} Hz", (gamma_peak_freq, Pn[g_idx]),
                textcoords="offset points", xytext=(8, -12), color='r')
    ax.set_xlim(0, 150)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB, norm)")
    ax.set_title(f"PSTH spectrum  (gamma peak {gamma_peak_freq:.1f} Hz, "
                 f"Gamma power {gamma_ratio:.0f}%)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    twin = (t_rec >= t_transient) & (t_rec <= t_transient + 300.0)
    for k in range(4):
        ax.plot(t_rec[twin], V_rec[twin, k] + k * 100, lw=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV, offset)")
    ax.set_title("Sample membrane potentials (4 neurons)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Wang-Buzsaki 1996 ING @ GeNN  |  N={N}, Msyn={Msyn}, "
                 f"w=0.1/Msyn={w_per_syn:.4f}, Iapp={args.iapp}"
                 f"{'+'+str(args.iapp_std) if args.iapp_std>0 else ''}  |  "
                 f"rate={mean_rate:.1f} Hz, kappa={kappa:.2f}, "
                 f"gamma peak={gamma_peak_freq:.1f} Hz",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = os.path.join(outdir, f"wb1996_genn_{tag}.png")
    fig.savefig(png_path, dpi=130)
    print(f"图已保存: {png_path}")

    # ---- 保存数据 ----
    npz_path = os.path.join(outdir, f"wb1996_genn_{tag}.npz")
    np.savez_compressed(npz_path, spk_t=spk_t, spk_id=spk_id,
                        V_rec=V_rec, t_rec=t_rec,
                        pop_rate=pop_rate, t_psth=t_psth,
                        f_ax=f_ax, P=P,
                        params=dict(N=N, Msyn=Msyn, w_per_syn=w_per_syn,
                                    alpha_syn=alpha_syn, beta_syn=beta_syn,
                                    E_syn=E_syn, iapp=args.iapp,
                                    iapp_std=args.iapp_std, dt=args.dt,
                                    T=args.T, seed=args.seed),
                        metrics=dict(mean_rate=mean_rate, std_rate=std_rate,
                                     peak_freq=peak_freq,
                                     gamma_peak_freq=gamma_peak_freq,
                                     gamma_ratio=gamma_ratio, kappa=kappa))
    print(f"数据已保存: {npz_path}")


if __name__ == "__main__":
    main()
