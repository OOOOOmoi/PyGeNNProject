"""
Probabilistic decision making by slow reverberation in cortical circuits.
X-J Wang, Neuron 2002.

http://dx.doi.org/10.1016/S0896-6273(02)01092-9

MPI/multiprocessing version — converted to MAM_MPI framework.

Architecture (matching MAM_MPI technical approach):
  - 4 "areas": E0 (non-selective E), E1 (selective-1), E2 (selective-2), I
  - Each area assigned to a worker process on a specific GPU
  - Master-worker Queue communication for spike count exchange
  - Host-side population-level trace computation (replaces per-neuron traces)
    * trace_sum = trace_sum * exp(-batch_dt/tau) + batch_spike_count  (AMPA, GABA)
    * x_nmda_sum = x_nmda_sum * exp(-batch_dt/tau_x) + batch_spike_count
    * sNMDA_sum ≈ sNMDA_sum * (1 - batch_dt/tauNMDA) + alpha * batch_dt * x_nmda_sum
  - S_AMPA/S_NMDA/S_GABA computed from trace sums × W matrix, pushed to GPU
  - SpikeCount variable for efficient batched spike counting (no per-step recording pull)
  - Per-worker logging in MAM_MPI format
"""

import numpy as np
from argparse import ArgumentParser, Namespace
import pygenn
from pygenn import GeNNModel, VarLocation, init_var
from pygenn.cuda_backend import DeviceSelect
from time import perf_counter
import os
import time
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

# ============================================================================
# 常量
# ============================================================================
DT_MS = 0.02  # 时间步长 (ms) — Wang 2002 需要精细步长

# ============================================================================
# 自定义神经元模型 — 简化版 Wang 2002
# ============================================================================
# 与原版区别：
#   - 移除了 per-neuron 突触迹 (sAMPA, x_nmda, sNMDA, sGABA)
#   - S_AMPA, S_NMDA, S_GABA 改为从 host 推送 (READ_ONLY)
#   - 添加 SpikeCount 变量，在 reset_code 中递增，用于批量脉冲计数
#   - 外部 Poisson 背景用 sim_code 内 gennrand_uniform 实现
#   - reset_code 不再累加迹（迹由 host 端从 SpikeCount delta 计算）

wang2002_neuron_mpi = pygenn.create_neuron_model(
    "Wang2002MPI",
    params=[
        "C", "TauM", "Vrest", "Vreset", "Vthresh", "TauRefrac",
        "TauAMPA",
        "V_E", "V_I",
        "a_nmda", "b_nmda",
        "gAMPA_ext", "gAMPA", "gNMDA", "gGABA",
        "rate_ext",
    ],
    vars=[
        ("V",          "scalar", pygenn.VarAccess.READ_WRITE),
        ("RefracTime", "scalar", pygenn.VarAccess.READ_WRITE),
        ("sAMPA_ext",  "scalar", pygenn.VarAccess.READ_WRITE),
        ("SpikeCount", "scalar", pygenn.VarAccess.READ_WRITE),
        ("rate_stim",  "scalar", pygenn.VarAccess.READ_ONLY),
        ("S_AMPA",     "scalar", pygenn.VarAccess.READ_ONLY),
        ("S_NMDA",     "scalar", pygenn.VarAccess.READ_ONLY),
        ("S_GABA",     "scalar", pygenn.VarAccess.READ_ONLY),
    ],
    sim_code="""
    // --- Poisson 背景输入 ---
    if (gennrand_uniform() < rate_ext * dt) {
        sAMPA_ext += 1.0f;
    }
    // --- 刺激 Poisson（仅选择性群体） ---
    if (rate_stim > 0.0f && gennrand_uniform() < rate_stim * dt) {
        sAMPA_ext += 1.0f;
    }

    // --- LIF 动力学 ---
    if (RefracTime <= 0.0f) {
        const scalar I_leak = (C / TauM) * (V - Vrest);
        const scalar I_AMPA_ext = gAMPA_ext * sAMPA_ext * (V - V_E);
        const scalar I_AMPA     = gAMPA * S_AMPA * (V - V_E);
        const scalar I_NMDA     = gNMDA * S_NMDA * (V - V_E)
                                  / (1.0f + exp(-a_nmda * V) / b_nmda);
        const scalar I_GABA     = gGABA * S_GABA * (V - V_I);
        const scalar Isyn = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA;
        V += (-I_leak - Isyn) / C * dt;
    }
    else {
        RefracTime -= dt;
    }

    // --- 外部 AMPA 迹衰减 ---
    sAMPA_ext -= sAMPA_ext / TauAMPA * dt;
    """,
    threshold_condition_code="(RefracTime <= 0.0f) && (V >= Vthresh)",
    reset_code="""
    V = Vreset;
    RefracTime = TauRefrac;
    SpikeCount += 1.0f;
    """,
)


# ============================================================================
# 参数 — 与 Brian2 版本完全一致
# ============================================================================

modelparams = {
    "V_L": -70.0, "Vth": -50.0, "Vreset": -55.0,
    "gE": 0.025, "tau_m_E": 20.0, "tau_ref_E": 2.0,
    "gI": 0.020, "tau_m_I": 10.0, "tau_ref_I": 1.0,
    "V_E": 0.0, "V_I": -70.0,
    "a": 0.062, "b": 3.57,
    "tauAMPA": 2.0, "tau_x": 2.0, "tauNMDA": 100.0,
    "alpha": 0.5, "tauGABA": 5.0,
    "gAMPA_ext_E": 0.0021, "gAMPA_ext_I": 0.00162,
    "gAMPA_E_raw": 0.080, "gNMDA_E_raw": 0.264, "gGABA_E_raw": 0.520,
    "gAMPA_I_raw": 0.064, "gNMDA_I_raw": 0.208, "gGABA_I_raw": 0.400,
    "nu_ext": 2.4,
    "N_E": 1600, "N_I": 400,
    "fsel": 0.15, "wp": 1.7,
}


# ============================================================================
# 区域定义 — 4 个"区域"对应 MAM_MPI 中的 area 概念
# ============================================================================

def build_area_config(params):
    """
    构建 4 个区域配置，返回 area_list 和每个 area 的参数。

    区域映射:
      E0 → 非选择性兴奋神经元 (N0 = N_E - 2*N1)
      E1 → 选择性群体 1 (N1 = fsel * N_E)
      E2 → 选择性群体 2 (N2 = N1)
      I  → 抑制性神经元 (N_I)
    """
    N_E = params["N_E"]
    N_I = params["N_I"]
    fsel = params["fsel"]
    wp = params["wp"]

    N1 = int(fsel * N_E)
    N2 = N1
    N0 = N_E - N1 - N2

    wm = (1.0 - wp * fsel) / (1.0 - fsel)

    # W 矩阵: 行=目标, 列=源
    W = np.array([
        [1.0, 1.0, 1.0],
        [ wm,  wp,  wm],
        [ wm,  wm,  wp],
    ], dtype=np.float64)

    # 电导缩放
    gAMPA_E = params["gAMPA_E_raw"] / N_E
    gNMDA_E = params["gNMDA_E_raw"] / N_E
    gGABA_E = params["gGABA_E_raw"] / N_I
    gAMPA_I = params["gAMPA_I_raw"] / N_E
    gNMDA_I = params["gNMDA_I_raw"] / N_E
    gGABA_I = params["gGABA_I_raw"] / N_I

    C_E = params["gE"] * params["tau_m_E"]
    C_I = params["gI"] * params["tau_m_I"]

    # 每个 area 的神经元参数
    area_config = {
        "E0": {
            "N": N0, "is_inhib": False, "is_selective": 0,
            "params": {
                "C": C_E, "TauM": params["tau_m_E"],
                "Vrest": params["V_L"], "Vreset": params["Vreset"],
                "Vthresh": params["Vth"], "TauRefrac": params["tau_ref_E"],
                "TauAMPA": params["tauAMPA"],
                "V_E": params["V_E"], "V_I": params["V_I"],
                "a_nmda": params["a"], "b_nmda": params["b"],
                "gAMPA_ext": params["gAMPA_ext_E"],
                "gAMPA": gAMPA_E, "gNMDA": gNMDA_E, "gGABA": gGABA_E,
                "rate_ext": params["nu_ext"],
            },
        },
        "E1": {
            "N": N1, "is_inhib": False, "is_selective": 1,
            "params": {
                "C": C_E, "TauM": params["tau_m_E"],
                "Vrest": params["V_L"], "Vreset": params["Vreset"],
                "Vthresh": params["Vth"], "TauRefrac": params["tau_ref_E"],
                "TauAMPA": params["tauAMPA"],
                "V_E": params["V_E"], "V_I": params["V_I"],
                "a_nmda": params["a"], "b_nmda": params["b"],
                "gAMPA_ext": params["gAMPA_ext_E"],
                "gAMPA": gAMPA_E, "gNMDA": gNMDA_E, "gGABA": gGABA_E,
                "rate_ext": params["nu_ext"],
            },
        },
        "E2": {
            "N": N2, "is_inhib": False, "is_selective": 2,
            "params": {
                "C": C_E, "TauM": params["tau_m_E"],
                "Vrest": params["V_L"], "Vreset": params["Vreset"],
                "Vthresh": params["Vth"], "TauRefrac": params["tau_ref_E"],
                "TauAMPA": params["tauAMPA"],
                "V_E": params["V_E"], "V_I": params["V_I"],
                "a_nmda": params["a"], "b_nmda": params["b"],
                "gAMPA_ext": params["gAMPA_ext_E"],
                "gAMPA": gAMPA_E, "gNMDA": gNMDA_E, "gGABA": gGABA_E,
                "rate_ext": params["nu_ext"],
            },
        },
        "I": {
            "N": N_I, "is_inhib": True, "is_selective": 0,
            "params": {
                "C": C_I, "TauM": params["tau_m_I"],
                "Vrest": params["V_L"], "Vreset": params["Vreset"],
                "Vthresh": params["Vth"], "TauRefrac": params["tau_ref_I"],
                "TauAMPA": params["tauAMPA"],
                "V_E": params["V_E"], "V_I": params["V_I"],
                "a_nmda": params["a"], "b_nmda": params["b"],
                "gAMPA_ext": params["gAMPA_ext_I"],
                "gAMPA": gAMPA_I, "gNMDA": gNMDA_I, "gGABA": gGABA_I,
                "rate_ext": params["nu_ext"],
            },
        },
    }

    return area_config, W, N0, N1, N2


# ============================================================================
# 区域索引分配 — 类似 MAM_MPI 的 split_indices
# ============================================================================

def split_areas(area_list, num_workers):
    """将 area_list 均匀划分为 num_workers 份。"""
    n = len(area_list)
    chunk = (n + num_workers - 1) // num_workers
    return [area_list[i*chunk:(i+1)*chunk] for i in range(num_workers) if area_list[i*chunk:(i+1)*chunk]]


# ============================================================================
# 群体级突触迹计算 (host 端) — 批量衰减版本
# ============================================================================

class PopulationTraces:
    """
    维护群体级突触迹求和，用批量脉冲计数驱动。

    数学等价性 (批量近似):
      Σ_neurons sAMPA_neuron(t) ≈ exp(-batch_dt/tau) * Σ_prev + batch_spike_count

    对于 NMDA (小 sNMDA 近似):
      x_nmda_sum(t) = exp(-batch_dt/tau_x) * x_nmda_sum_prev + batch_spike_count
      sNMDA_sum(t) = sNMDA_sum_prev * (1 - batch_dt/tauNMDA) + alpha * batch_dt * x_nmda_sum(t)
    """

    def __init__(self, params):
        self.tauAMPA = params["tauAMPA"]
        self.tau_x = params["tau_x"]
        self.tauNMDA = params["tauNMDA"]
        self.alpha = params["alpha"]
        self.tauGABA = params["tauGABA"]

        # 3 个 E 子群的迹
        self.sAMPA = np.zeros(3, dtype=np.float64)  # E0, E1, E2
        self.x_nmda = np.zeros(3, dtype=np.float64)
        self.sNMDA = np.zeros(3, dtype=np.float64)
        # I 的迹
        self.sGABA = 0.0

    def update(self, spike_counts_E, spike_count_I, batch_dt):
        """
        用当前批量脉冲计数更新迹求和。

        Parameters
        ----------
        spike_counts_E : array-like of 3 ints
            E0, E1, E2 各自的批量脉冲计数
        spike_count_I : int
            I 的批量脉冲计数
        batch_dt : float
            批量时间间隔 (ms)
        """
        # AMPA 迹
        self.sAMPA = self.sAMPA * math.exp(-batch_dt / self.tauAMPA) + spike_counts_E
        # NMDA 门控变量
        self.x_nmda = self.x_nmda * math.exp(-batch_dt / self.tau_x) + spike_counts_E
        # NMDA 迹 (线性近似)
        self.sNMDA = self.sNMDA * (1.0 - batch_dt / self.tauNMDA) + self.alpha * batch_dt * self.x_nmda
        # GABA 迹
        self.sGABA = self.sGABA * math.exp(-batch_dt / self.tauGABA) + spike_count_I

    def compute_S(self, W):
        """
        用 W 矩阵计算群体级加权和。

        Returns
        -------
        dict: {
            "E0": {"S_AMPA": float, "S_NMDA": float, "S_GABA": float},
            "E1": {...}, "E2": {...},
            "I":  {"S_AMPA": float, "S_NMDA": float, "S_GABA": float},
        }
        """
        S_ampa = W.dot(self.sAMPA)  # shape (3,)
        S_nmda = W.dot(self.sNMDA)  # shape (3,)
        S_gaba = self.sGABA

        return {
            "E0": {"S_AMPA": S_ampa[0], "S_NMDA": S_nmda[0], "S_GABA": S_gaba},
            "E1": {"S_AMPA": S_ampa[1], "S_NMDA": S_nmda[1], "S_GABA": S_gaba},
            "E2": {"S_AMPA": S_ampa[2], "S_NMDA": S_nmda[2], "S_GABA": S_gaba},
            # I 接收与非选择性 E0 相同的兴奋性输入
            "I":  {"S_AMPA": S_ampa[0], "S_NMDA": S_nmda[0], "S_GABA": S_gaba},
        }


# ============================================================================
# 刺激协议
# ============================================================================

class Stimulus:
    def __init__(self, Ton, Toff, mu0, coh):
        self.Ton = Ton
        self.Toff = Toff
        self.mu0 = mu0
        self.set_coh(coh)

    def set_coh(self, coh):
        self.pos_rate = self.mu0 * (1.0 + coh / 100.0)
        self.neg_rate = self.mu0 * (1.0 - coh / 100.0)

    def get_rate(self, t, selective_id):
        """selective_id: 1 for E1, 2 for E2, 0 for no stimulus."""
        if self.Ton <= t < self.Toff:
            if selective_id == 1:
                return self.pos_rate
            elif selective_id == 2:
                return self.neg_rate
        return 0.0


# ============================================================================
# Worker 进程函数 — 类似 MAM_MPI 的 Part()
# ============================================================================

def Part(worker_id, gpu_id, assigned_areas, area_config, W,
         stim_params, model_name, args,
         to_master, from_master, done_queue, final_queue):
    """
    每个 worker 负责 1+ 个区域，在指定 GPU 上运行。

    通信协议 (与 MAM_MPI 一致):
      1. 运行 batch_steps 步 → 拉取 SpikeCount → 计算 delta → 发送到主进程
      2. done_queue.put(worker_id) 通知主进程
      3. from_master.get() 等待主进程聚合后的 S_* 数据
      4. 推送 S_* 到 GPU → 下一批

    SpikeCount 机制:
      - 神经元模型在 reset_code 中递增 SpikeCount
      - 每批结束后 pull SpikeCount，计算与上一批的 delta
      - 避免每步 pull recording buffer 的巨大 GPU sync 开销
    """
    batch_steps = args.batch_steps
    print(f"[Worker {worker_id}] 启动, GPU={gpu_id}, areas={assigned_areas}, batch_steps={batch_steps}")

    # ---- 创建 GeNN 模型 ----
    model = GeNNModel("float", f"GenCODE/worker{worker_id}_on_device{gpu_id}",
                      device_select_method=DeviceSelect.MANUAL, manual_device_id=gpu_id)
    model.dt = DT_MS
    model.fuse_postsynaptic_models = False
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    model.timing_enabled = True

    # ---- 创建神经元群体 ----
    neuron_pops = {}
    total_neurons = 0
    var_init = {
        "V": init_var("Uniform", {"min": modelparams["Vreset"], "max": modelparams["Vth"]}),
        "RefracTime": 0.0,
        "sAMPA_ext": 0.0,
        "SpikeCount": 0.0,
        "rate_stim": 0.0,
        "S_AMPA": 0.0, "S_NMDA": 0.0, "S_GABA": 0.0,
    }

    for area_name in assigned_areas:
        cfg = area_config[area_name]
        n = cfg["N"]
        if n <= 0:
            continue
        pop = model.add_neuron_population(
            f"pop_{area_name}", n,
            wang2002_neuron_mpi, cfg["params"], var_init,
        )
        pop.spike_recording_enabled = True
        neuron_pops[area_name] = pop
        total_neurons += n

    print(f"[Worker {worker_id}] 总神经元数: {total_neurons}")

    # ---- 编译和加载 ----
    # recording buffer 设为整个仿真长度，仅在结束时 pull 一次用于绘图
    t0 = perf_counter()
    model.build()
    build_time = perf_counter() - t0

    t0 = perf_counter()
    duration_steps = int(round(args.duration / DT_MS))
    model.load(num_recording_timesteps=duration_steps)
    load_time = perf_counter() - t0
    print(f"[Worker {worker_id}] build={build_time:.2f}s, load={load_time:.2f}s")

    # ---- 日志 ----
    os.makedirs(f"log/log_{model_name}", exist_ok=True)
    log_path = f"log/log_{model_name}/worker_{worker_id}.log"
    open(log_path, "w").close()
    with open(log_path, "a") as f:
        f.write(f"{total_neurons},{len(neuron_pops)},0,0,"
                f"{build_time*1000:.2f},{load_time*1000:.2f}\n")

    # ---- 刺激 ----
    stimulus = Stimulus(stim_params["Ton"], stim_params["Toff"],
                        stim_params["mu0"], stim_params["coh"])

    # ---- 推送初始刺激速率 ----
    for area_name, pop in neuron_pops.items():
        sel = area_config[area_name]["is_selective"]
        rate = stimulus.get_rate(0.0, sel)
        pop.vars["rate_stim"].view[:] = rate
        pop.vars["rate_stim"].push_to_device()

    # ---- 初始化 SpikeCount 跟踪 ----
    prev_spike_counts = {}
    for area_name, pop in neuron_pops.items():
        prev_spike_counts[area_name] = 0.0

    # ---- 仿真主循环 (批量模式) ----
    current_batch = 0
    log_buffer = []
    log_flush_interval = 200  # 每 200 批写一次日志
    stim_on = False
    Ton = stimulus.Ton
    Toff = stimulus.Toff

    total_batches = (duration_steps + batch_steps - 1) // batch_steps
    batch_dt = batch_steps * DT_MS

    t_start = perf_counter()

    while model.t < args.duration:
        # ================================================================
        # 1) 运行 batch_steps 步 (GPU 自治运行, 使用当前 S_* 值)
        # ================================================================
        t_batch0 = perf_counter()

        for _ in range(batch_steps):
            if model.t >= args.duration:
                break

            model.step_time()
            t = model.t

            # 刺激边界检测
            is_in_window = (Ton <= t < Toff)
            if not stim_on and is_in_window:
                stim_on = True
                for area_name, pop in neuron_pops.items():
                    sel = area_config[area_name]["is_selective"]
                    rate = stimulus.get_rate(t, sel)
                    pop.vars["rate_stim"].view[:] = rate
                    pop.vars["rate_stim"].push_to_device()
                print(f"[Worker {worker_id}] Stimulus ON at t={t:.1f} ms")
            elif stim_on and not is_in_window:
                stim_on = False
                for area_name, pop in neuron_pops.items():
                    pop.vars["rate_stim"].view[:] = 0.0
                    pop.vars["rate_stim"].push_to_device()
                print(f"[Worker {worker_id}] Stimulus OFF at t={t:.1f} ms")

        t_batch1 = perf_counter()

        # ================================================================
        # 2) 拉取 SpikeCount, 计算本批量脉冲计数 (delta)
        # ================================================================
        t_count0 = perf_counter()
        spike_counts = {}
        for area_name, pop in neuron_pops.items():
            pop.vars["SpikeCount"].pull_from_device()
            current_count = float(np.sum(pop.vars["SpikeCount"].view[:]))
            spike_counts[area_name] = int(current_count - prev_spike_counts[area_name])
            prev_spike_counts[area_name] = current_count
        t_count1 = perf_counter()

        # ================================================================
        # 3) 发送脉冲计数给主进程
        # ================================================================
        msg = {
            "worker_id": worker_id,
            "spike_counts": spike_counts,
            "timestamp": time.perf_counter(),
        }
        to_master.put(msg)
        done_queue.put(worker_id)

        # ================================================================
        # 4) 等待主进程回复 S_*
        # ================================================================
        reply = from_master.get()
        if reply["type"] == "stop":
            break

        S_data = reply["S_values"]

        # ================================================================
        # 5) 推送 S_AMPA/S_NMDA/S_GABA 到 GPU
        # ================================================================
        t_push0 = perf_counter()
        for area_name, pop in neuron_pops.items():
            if area_name in S_data:
                sv = S_data[area_name]
                pop.vars["S_AMPA"].view[:] = sv["S_AMPA"]
                pop.vars["S_NMDA"].view[:] = sv["S_NMDA"]
                pop.vars["S_GABA"].view[:] = sv["S_GABA"]
                pop.vars["S_AMPA"].push_to_device()
                pop.vars["S_NMDA"].push_to_device()
                pop.vars["S_GABA"].push_to_device()
        t_push1 = perf_counter()

        # ================================================================
        # 6) 日志 (MAM_MPI 风格)
        # ================================================================
        log_buffer.append(
            f"{model.timestep},"
            f"{(t_batch1 - t_batch0) * 1000:.3f},"      # batch_step_time
            f"{(t_count1 - t_count0) * 1000:.3f},"       # spike_count
            f"{(t_push1 - t_push0) * 1000:.3f},"         # push_S
            f"{(t_push1 - t_batch0) * 1000:.3f}"         # total_batch
        )
        if len(log_buffer) >= log_flush_interval:
            with open(log_path, "a") as f:
                f.write("\n".join(log_buffer) + "\n")
            log_buffer.clear()

        current_batch += 1
        if current_batch % 200 == 0:
            elapsed = perf_counter() - t_start
            pct = 100.0 * model.t / args.duration
            print(f"[Worker {worker_id}] {pct:.1f}% ({model.t:.0f}/{args.duration:.0f} ms) "
                  f"elapsed {elapsed:.1f}s")

    # ---- 写入总时间 ----
    total_time = perf_counter() - t_start
    with open(log_path, "a") as f:
        if log_buffer:
            f.write("\n".join(log_buffer) + "\n")
        f.write(f"total_simulation_time,{total_time * 1000:.2f}\n")

    print(f"[Worker {worker_id}] 仿真完成, 耗时 {total_time:.2f}s")

    # ---- 最终拉取 recording buffer 用于绘图 ----
    model.pull_recording_buffers_from_device()
    final_spike_data = {}
    for area_name, pop in neuron_pops.items():
        spike_times, spike_ids = pop.spike_recording_data[0]
        if len(spike_times) > 0:
            final_spike_data[area_name] = np.column_stack((spike_times, spike_ids))
        else:
            final_spike_data[area_name] = np.empty((0, 2))

    # ---- 发送最终脉冲数据给主进程 ----
    final_queue.put({
        "worker_id": worker_id,
        "spike_data": final_spike_data,
        "areas": list(neuron_pops.keys()),
    })


# ============================================================================
# 主进程 — 类似 MAM_MPI 的主循环
# ============================================================================

def main():
    args = parse_args()
    duration = args.duration
    model_name = f"wang2002_mpi_coh{args.coh}_seed{args.seed}"

    # 构建区域配置
    area_config, W, N0, N1, N2 = build_area_config(modelparams)
    area_list = list(area_config.keys())  # ["E0", "E1", "E2", "I"]

    num_workers = min(args.num_workers, len(area_list))
    gpu_ids = args.gpu_ids
    area_splits = split_areas(area_list, num_workers)

    batch_steps = args.batch_steps
    batch_dt = batch_steps * DT_MS

    print(f"Wang 2002 MPI — {num_workers} workers, {len(gpu_ids)} GPU(s)")
    print(f"Areas: {area_list}")
    print(f"Split: {area_splits}")
    print(f"W matrix:\n{W}")
    print(f"N0={N0}, N1={N1}, N2={N2}, N_I={modelparams['N_I']}")
    print(f"batch_steps={batch_steps}, batch_dt={batch_dt:.3f} ms")
    print(f"Total steps={int(duration/DT_MS)}, Total batches={int(duration/DT_MS/batch_steps)}")

    # 刺激参数
    stim_params = {
        "Ton": args.stim_on,
        "Toff": args.stim_off,
        "mu0": args.mu0,
        "coh": args.coh,
    }

    # ---- 创建通信队列 ----
    to_master_queues = []
    from_master_queues = []
    processes = []
    done_queue = Queue()
    final_queue = Queue()

    for i in range(num_workers):
        to_master = Queue()
        from_master = Queue()
        to_master_queues.append(to_master)
        from_master_queues.append(from_master)
        gpu_id = gpu_ids[i % len(gpu_ids)]
        assigned = area_splits[i]
        p = Process(target=Part,
                    args=(i, gpu_id, assigned, area_config, W,
                          stim_params, model_name, args,
                          to_master, from_master, done_queue, final_queue))
        p.start()
        processes.append(p)

    # ---- 主循环: 聚合脉冲计数, 计算 S_*, 广播 ----
    traces = PopulationTraces(modelparams)
    max_batches = int(duration / DT_MS / batch_steps) + 1
    master_log = []
    batch_num = 0

    os.makedirs(f"log/log_{model_name}", exist_ok=True)
    print(f"主循环开始: {max_batches} batches, batch_dt={batch_dt:.3f} ms")

    while batch_num < max_batches:
        per_worker_counts = {}

        # 等待所有 worker 提交脉冲计数
        for _ in range(num_workers):
            wid = done_queue.get()
            msg = to_master_queues[wid].get()
            per_worker_counts[msg["worker_id"]] = msg["spike_counts"]

        # 聚合各区域的脉冲计数
        agg_counts = {"E0": 0, "E1": 0, "E2": 0, "I": 0}
        for wid, counts in per_worker_counts.items():
            for area, cnt in counts.items():
                agg_counts[area] = agg_counts.get(area, 0) + cnt

        # 更新群体级迹 (批量衰减)
        spike_counts_E = np.array([agg_counts["E0"], agg_counts["E1"], agg_counts["E2"]])
        spike_count_I = agg_counts["I"]
        traces.update(spike_counts_E, spike_count_I, batch_dt)

        # 计算 S_*
        S_values = traces.compute_S(W)

        # 广播给所有 worker
        for q in from_master_queues:
            q.put({"type": "continue", "S_values": S_values})

        # 主进程日志
        master_log.append(
            f"{batch_num},{agg_counts['E0']},{agg_counts['E1']},{agg_counts['E2']},{agg_counts['I']}"
        )
        if len(master_log) >= 500:
            with open(f"log/log_{model_name}/master.log", "a") as f:
                f.write("\n".join(master_log) + "\n")
            master_log.clear()

        batch_num += 1

        if batch_num % 500 == 0:
            print(f"[主进程] batch {batch_num}/{max_batches}, "
                  f"E0={agg_counts['E0']} E1={agg_counts['E1']} "
                  f"E2={agg_counts['E2']} I={agg_counts['I']}")

    # ---- 发送停止信号 ----
    for q in from_master_queues:
        q.put({"type": "stop"})

    # ---- 收集最终脉冲数据 ----
    final_data = {}
    for _ in range(num_workers):
        msg = final_queue.get()
        print(f"主进程收到 worker {msg['worker_id']} 的最终数据 "
              f"({sum(len(v) for v in msg['spike_data'].values())} spikes)")
        for area in msg["areas"]:
            data = msg["spike_data"][area]
            if len(data) > 0:
                final_data[area] = data

    # ---- 等待子进程退出 ----
    for p in processes:
        p.join()

    # 刷新剩余日志
    if master_log:
        with open(f"log/log_{model_name}/master.log", "a") as f:
            f.write("\n".join(master_log) + "\n")

    print("所有子进程已结束。")

    # ---- 保存和绘图 ----
    save_and_plot(final_data, model_name, args, N0, N1, N2)

    # ---- 保存迹数据 ----
    save_traces(traces, model_name)


# ============================================================================
# 保存和绘图
# ============================================================================

def save_and_plot(final_data, model_name, args, N0, N1, N2):
    """保存脉冲数据并生成图表。"""
    os.makedirs("output", exist_ok=True)

    # 合并 E 子群
    spikes_E = []
    spikes_I = []

    for area, data in final_data.items():
        if area.startswith("E"):
            # 偏移神经元 ID 以匹配原始 E 群体编号
            offset = {"E0": 0, "E1": N0, "E2": N0 + N1}[area]
            if len(data) > 0:
                data[:, 1] += offset
                spikes_E.append(data)
        elif area == "I":
            if len(data) > 0:
                spikes_I.append(data)

    if spikes_E:
        spikes_E = np.vstack(spikes_E)
        sort_idx = np.argsort(spikes_E[:, 0])
        spikes_E = spikes_E[sort_idx]
    else:
        spikes_E = np.empty((0, 2))

    if spikes_I:
        spikes_I = np.vstack(spikes_I)
        sort_idx = np.argsort(spikes_I[:, 0])
        spikes_I = spikes_I[sort_idx]
    else:
        spikes_I = np.empty((0, 2))

    # 保存脉冲数据
    np.savetxt(f"output/spikesE_{model_name}.txt", spikes_E,
               fmt="%-9d %25.18e",
               header="{:<8} {:<25}".format("Neuron", "Time (ms)"))
    np.savetxt(f"output/spikesI_{model_name}.txt", spikes_I,
               fmt="%-9d %25.18e",
               header="{:<8} {:<25}".format("Neuron", "Time (ms)"))

    # 统计
    n_E = len(spikes_E)
    n_I = len(spikes_I)
    rate_E = n_E / modelparams["N_E"] / (args.duration * 0.001)
    rate_I = n_I / modelparams["N_I"] / (args.duration * 0.001)
    print(f"E: {n_E} spikes, avg rate {rate_E:.1f} Hz")
    print(f"I: {n_I} spikes, avg rate {rate_I:.1f} Hz")

    # ---- Raster plot ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    if len(spikes_E) > 0:
        axes[0].scatter(spikes_E[:, 0] * 0.001, spikes_E[:, 1],
                        s=0.8, c="black", marker=".", rasterized=True)
    axes[0].set_ylabel("Neuron index (E)")
    axes[0].set_title(f"Wang 2002 — MPI (coh={args.coh}, seed={args.seed})")
    for y, lbl, c in [(N0, f"N0={N0}", "blue"),
                      (N0 + N1, f"N1={N1}", "red"),
                      (N0 + N1 + N2, f"N2={N2}", "orange")]:
        axes[0].axhline(y=y, color=c, ls="--", alpha=0.4)
        axes[0].text(0.01, y + 5, lbl, color=c, fontsize=7, va="bottom")

    if len(spikes_I) > 0:
        axes[1].scatter(spikes_I[:, 0] * 0.001, spikes_I[:, 1],
                        s=0.8, c="red", marker=".", rasterized=True)
    axes[1].set_ylabel("Neuron index (I)")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"output/{model_name}_raster.pdf", dpi=150)
    print(f"Raster → output/{model_name}_raster.pdf")
    plt.close()

    # ---- Firing rates ----
    bin_ms = 5.0
    t_end = args.duration
    bins = np.arange(0, t_end + bin_ms, bin_ms)
    t_centers = (bins[:-1] + bins[1:]) * 0.0005

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for lbl, lo, hi, c in [("Non-sel E0", 0, N0, "gray"),
                           ("Sel-1 E1", N0, N0 + N1, "blue"),
                           ("Sel-2 E2", N0 + N1, N0 + N1 + N2, "red")]:
        if len(spikes_E) > 0:
            m = (spikes_E[:, 1] >= lo) & (spikes_E[:, 1] < hi)
            if m.any():
                h, _ = np.histogram(spikes_E[m, 0], bins)
                axes[0].plot(t_centers, h / (hi - lo) / (bin_ms * 0.001),
                             label=lbl, color=c, lw=1)
    axes[0].set_ylabel("Rate (Hz)")
    axes[0].set_title("E Subpopulation Firing Rates")
    axes[0].legend(fontsize=7)

    if len(spikes_I):
        hI, _ = np.histogram(spikes_I[:, 0], bins)
        axes[1].plot(t_centers, hI / modelparams["N_I"] / (bin_ms * 0.001),
                     color="green", lw=1)
    axes[1].set_ylabel("Rate (Hz)")
    axes[1].set_title("Inhibitory Population")

    if len(spikes_E) > 0:
        hE, _ = np.histogram(spikes_E[:, 0], bins)
        axes[2].plot(t_centers, hE / modelparams["N_E"] / (bin_ms * 0.001),
                     label="E avg", color="black", lw=1)
    if len(spikes_I):
        axes[2].plot(t_centers, hI / modelparams["N_I"] / (bin_ms * 0.001),
                     label="I avg", color="green", lw=1)
    axes[2].set_ylabel("Rate (Hz)")
    axes[2].set_title("Population-Averaged Rates")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(f"output/{model_name}_rates.pdf", dpi=150)
    print(f"Rates → output/{model_name}_rates.pdf")
    plt.close()


def save_traces(traces, model_name):
    """保存最终迹值用于调试。"""
    with open(f"log/log_{model_name}/traces_final.txt", "w") as f:
        f.write(f"sAMPA: {traces.sAMPA}\n")
        f.write(f"x_nmda: {traces.x_nmda}\n")
        f.write(f"sNMDA: {traces.sNMDA}\n")
        f.write(f"sGABA: {traces.sGABA}\n")


# ============================================================================
# 命令行参数
# ============================================================================

def parse_args():
    parser = ArgumentParser(description="Wang 2002 Decision-Making Model (MPI)")
    parser.add_argument("--duration", type=float, default=2000.0,
                        help="Simulation duration (ms)")
    parser.add_argument("--coh", type=float, default=51.2,
                        help="Percent coherence")
    parser.add_argument("--seed", type=int, default=4,
                        help="Random seed")
    parser.add_argument("--stim-on", type=float, default=500.0,
                        help="Stimulus onset (ms)")
    parser.add_argument("--stim-off", type=float, default=1500.0,
                        help="Stimulus offset (ms)")
    parser.add_argument("--mu0", type=float, default=0.040,
                        help="Base stimulus rate (kHz)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes (max 4 for Wang 2002)")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=[0],
                        help="GPU IDs to use")
    parser.add_argument("--batch-steps", type=int, default=20,
                        help="Steps per batch (controls trace update frequency, "
                             "default=20 → 0.4ms update interval)")
    args, _ = parser.parse_known_args()
    return args


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    main()
