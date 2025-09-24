# hmam_mpi.py
import os
import sys
import time
import uuid
import numpy as np
from time import perf_counter
from itertools import product
from collections import defaultdict

import pickle  # 用于估算大小（可选）
from mpi4py import MPI

# 你的原有 imports（保留）
from config import expLIF_dict, input, layer_map, vis_content, \
    get_NN, get_SN, get_weight, get_weight_ext, externalRates, get_cc_delay, \
    getModelName, remove_dash_from_index_columns, get_ext_rate, net
from visual import visualize
from record import record_spike

import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from pygenn.cuda_backend import DeviceSelect
from scipy.stats import norm
import pandas as pd

# ---- simulation settings (跟你原来一致) ----
NUM_THREADS_PER_SPIKE = 1
duration = 1000
DT_MS = 0.1
duration_timesteps = int(round(duration / DT_MS))
ten_percent_timestep = duration_timesteps // 10
buffer_size = 1   # 你想每步通信就设为1；真实运行建议大一些以减少通信频率

# ---- MPI init ----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # 包含 master
num_workers = size - 1
if num_workers <= 0:
    raise RuntimeError("需要至少 2 个 MPI 进程 (1 master + >=1 worker) 来运行。")

# 假定每台机器 GPU 数相同（例如 8）
GPUS_PER_NODE = 8

# ---- helper functions ----
def split_indices_for_workers(area_list, num_workers):
    """把 area_list 均分给 num_workers 个 worker。
       返回 list of lists: worker_idx 0 对应 MPI rank 1（第一个 worker）。
    """
    n = len(area_list)
    chunk_size = (n + num_workers - 1) // num_workers
    return [area_list[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]

def merge_spike_data(spike_data_blocks):
    merged = {}
    for block in spike_data_blocks:
        if block is None:
            continue
        for area, pop_dict in block.items():
            merged.setdefault(area, {})
            for pop, spikes in pop_dict.items():
                merged[area].setdefault(pop, [])
                merged[area][pop].extend(spikes)
    return merged

def merge_nn_data(NeuronNumber_blocks):
    merged = {}
    for block in NeuronNumber_blocks:
        if block is None:
            continue
        for area, pop_dict in block.items():
            merged.setdefault(area, {})
            for pop, count in pop_dict.items():
                merged[area].setdefault(pop, 0)
                merged[area][pop] += count
    return merged

# ---- Load static config once on all processes (cheaper than re-importing per worker) ----
area_list_all = net["area_list"]
area_list_all = [s.replace("-", "") for s in area_list_all]
layer_list = net["layer_list"]
pop_list = net["population_list"]

# Precompute NN, SN, weight etc. (all ranks will load them - ok)
NN = get_NN()
NN = remove_dash_from_index_columns(NN)
SN, SN_ext = get_SN()
SN = remove_dash_from_index_columns(SN)
SN_ext = remove_dash_from_index_columns(SN_ext)
rate_ext = get_ext_rate()
rate_ext = remove_dash_from_index_columns(rate_ext)
weight, weight_sd = get_weight()
weight = remove_dash_from_index_columns(weight)
weight_sd = remove_dash_from_index_columns(weight_sd)
delay_cc, delay_cc_sd = get_cc_delay()
delay_cc = remove_dash_from_index_columns(delay_cc)
delay_cc_sd = remove_dash_from_index_columns(delay_cc_sd)
weight_ext, weight_ext_sd = get_weight_ext()
weight_ext = remove_dash_from_index_columns(weight_ext)
weight_ext_sd = remove_dash_from_index_columns(weight_ext_sd)

# Precompute NeuronNumber global mapping for final visualize
NeuronNumber_global = defaultdict(dict)
for area in area_list_all:
    for layer in layer_list:
        for pop in pop_list:
            if (area, layer, pop) in NN.index:
                popNum = NN.loc[(area, layer, pop)]
                NeuronNumber_global[area][pop + layer_map[layer]] = popNum

# ---- MASTER (rank 0) ----
if rank == 0:
    print(f"[MASTER] rank=0 starting. total workers = {num_workers}")
    # decide mapping of area->worker
    splits = split_indices_for_workers(area_list_all, num_workers)
    # For debug:
    for i, s in enumerate(splits, start=1):
        print(f"[MASTER] worker_rank={i} assigned areas: {s}")

    max_steps = duration_timesteps // buffer_size
    step = 0
    all_steps_spike_data = []   # 存每一轮合并结果（用于最后合并/可视化）

    try:
        while step < max_steps:
            step += 1
            # 1) 收集来自任意 worker 的数据，直到收到 num_workers 条（tag = step）
            received = 0
            round_blocks = []
            recv_start = time.perf_counter()
            while received < num_workers:
                data = comm.recv(source=MPI.ANY_SOURCE, tag=step)  # data 是 worker 发送的 dict
                recv_time = perf_counter()
                # data should contain keys: worker_rank, spike_data, timestamp_sent
                worker_rank = data.get("worker_rank")
                ts_sent = data.get("timestamp_sent", None)
                # 计算延迟（注意：包含 worker 内序列化时间）
                if ts_sent is not None:
                    latency = recv_time - ts_sent
                else:
                    latency = None
                # 可选打印
                print(f"[MASTER][step {step}] recv from worker {worker_rank} latency={latency*1000:.3f} ms" if latency else f"[MASTER][step {step}] recv from worker {worker_rank}")
                round_blocks.append(data.get("spike_data"))
                received += 1

            # 2) 合并并统计（按你原有逻辑）
            all_spike_data = merge_spike_data(round_blocks)
            all_steps_spike_data.append(all_spike_data)

            processed_data = {"rate": {}, "spike_count": {}}
            for area, pop_dict in all_spike_data.items():
                processed_data["rate"].setdefault(area, {})
                processed_data["spike_count"].setdefault(area, {})
                for pop, data_chunks in pop_dict.items():
                    if not data_chunks:
                        processed_data["rate"][area][pop] = 0.0
                        processed_data["spike_count"][area][pop] = 0
                        continue
                    all_spikes = np.vstack(data_chunks)
                    spike_count = all_spikes.shape[0]
                    num_neurons = NeuronNumber_global[area][pop]
                    spike_rate = spike_count / num_neurons * 1000  # Hz over buffer (since buffer_size may be 1, 使用该近似)
                    processed_data["rate"][area][pop] = spike_rate
                    processed_data["spike_count"][area][pop] = spike_count
                    # 可选打印
                    print(f"[MASTER][step {step}] {area} {pop} spike_rate={spike_rate:.3f}Hz spike_count={spike_count}")

            # 3) 广播 update 给所有 worker（使用 tag = step）
            control_msg = {
                "type": "continue" if step < max_steps else "stop",
                "send_tag": str(uuid.uuid4()),
                "updates": processed_data,
                "timestamp_master_send": perf_counter()
            }
            for w in range(1, size):
                comm.send(control_msg, dest=w, tag=step)

            # 4) 等待来自每个 worker 的 ack（tag = step + max_steps + 1 或可重用 step）
            acks = 0
            while acks < num_workers:
                ack = comm.recv(source=MPI.ANY_SOURCE, tag=step+max_steps+1)  # tag 区分阶段
                worker_rank = ack.get("worker_rank")
                worker_recv_time = ack.get("worker_recv_time")  # worker 收到 master 的时间点
                # master->worker latency ~ worker_recv_time - control_msg['timestamp_master_send']
                master_send_time = control_msg["timestamp_master_send"]
                latency_m2w = worker_recv_time - master_send_time
                print(f"[MASTER][step {step}] ack from {worker_rank} master->worker latency {latency_m2w*1000:.3f} ms")
                acks += 1

            # loop continue until step == max_steps
        # while end

        # ---- simulation finished on master side ----
        print("[MASTER] main loop complete. Sending final stop (if needed) and collecting final acks...")

        # ensure all workers got stop: we already sent 'stop' on last iter, but safe to re-send
        for w in range(1, size):
            # send a stop control with a unique tag to allow acks collection
            control_msg = {"type": "stop", "send_tag": str(uuid.uuid4()), "updates": {}, "timestamp_master_send": perf_counter()}
            comm.send(control_msg, dest=w, tag=max_steps+1)

        # collect final acks
        final_acks = 0
        while final_acks < num_workers:
            ack = comm.recv(source=MPI.ANY_SOURCE, tag=max_steps+2)
            final_acks += 1

        # ---- 合并所有时间步数据并可视化 ----
        final_spike_data = merge_spike_data(all_steps_spike_data)
        print("[MASTER] total areas in final_spike_data:", list(final_spike_data.keys()))
        # 可视化（在 master 上）
        visualize(suffix="mpi_test", spike_data=final_spike_data, duration=duration, model_name="HMAM", NeuronNumber=NeuronNumber_global)

    except Exception as e:
        print("[MASTER] 捕获异常：", e)
        raise

# ---- WORKER (rank > 0) ----
else:
    worker_rank = rank
    # Determine local device ID. We assume GPUS_PER_NODE GPUs per node and mpirun assigned ranks per node sequentially.
    # local_gpu_id = (rank-1) % GPUS_PER_NODE
    local_gpu_id = (rank - 1) % GPUS_PER_NODE
    print(f"[WORKER {worker_rank}] starting, local_gpu_id={local_gpu_id}")

    # decide which areas this worker is responsible for
    splits = split_indices_for_workers(area_list_all, num_workers)
    assigned_areas = splits[worker_rank - 1]  # worker rank 1 maps to splits[0]
    print(f"[WORKER {worker_rank}] assigned areas: {assigned_areas}")

    # Build model on assigned GPU (几乎照搬你原来的 Part)
    model = GeNNModel("float", f"HMAM_MPI_CODE/model_on_device{worker_rank}", device_select_method=DeviceSelect.MANUAL, manual_device_id=local_gpu_id)
    model.dt = DT_MS
    lif_init = {"V": init_var("Uniform", {"max": -50.0, "min": -200.0}), "RefracTime": 0.0}
    poisson_init = {"current": 0.0}

    NeuronNumber_local = defaultdict(dict)
    neuron_populations = defaultdict(dict)
    total_neurons = 0

    # create populations for assigned areas (简化复制你原有逻辑)
    for area in assigned_areas:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in NN.index:
                    popName = area + pop + layer_map[layer]
                    popNum = NN.loc[(area, layer, pop)]
                    NeuronNumber_local[area][pop + layer_map[layer]] = popNum
                    if popNum != 0:
                        if pop == "E":
                            neuronParam = net['neuron_params_E']
                        else:
                            neuronParam = net['neuron_params_I']
                        params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                                  "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                                  "Vthresh": neuronParam['V_th'], "Ioffset": 0,
                                  "TauRefrac": neuronParam['t_ref']}
                        neuron_pop = model.add_neuron_population(popName, popNum, "LIF", params, lif_init)
                        ext_weight = weight_ext.loc[(area, layer, pop)]
                        rate = rate_ext.loc[(area, layer, pop)] * 100
                        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                        model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)
                        neuron_pop.spike_recording_enabled = True
                        total_neurons += popNum
                        neuron_populations[area][pop + layer_map[layer]] = neuron_pop

    # synapses (省略细节复写你原来代码中的 loop)
    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 2})
    inh_curr_init = init_postsynaptic("ExpCurr", {"tau": 5})
    syn_group_num = 0
    total_synapses = 0
    for tar_area, src_area in product(assigned_areas, assigned_areas):
        for tar_layer, src_layer in product(layer_list, layer_list):
            for tar_pop, src_pop in product(pop_list, pop_list):
                tar = (tar_area, tar_layer, tar_pop)
                src = (src_area, src_layer, src_pop)
                if tar in SN.index and src in SN.columns:
                    synNum = SN.loc[tar, src]
                    if synNum <= 0:
                        continue
                    wAve = weight.loc[tar, src] / 1000
                    wSd = wAve / 10 / 1000
                    if src_area == tar_area:
                        if src_pop == 'E':
                            meanDelay = net['delay_e']; delay_sd = net['delay_e_sd']
                        else:
                            meanDelay = net['delay_i']; delay_sd = net['delay_i_sd']
                    else:
                        meanDelay = delay_cc.loc[(src_area, tar_area)]
                        delay_sd = delay_cc_sd.loc[(src_area, tar_area)]
                    quantile = 0.9999
                    normal_quantile_cdf = norm.ppf(quantile)
                    max_delay = meanDelay + (delay_sd * normal_quantile_cdf)
                    connect_params = {"num": synNum}
                    d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_delay}
                    total_synapses += synNum
                    syn_group_num += 1
                    matrix_type = "PROCEDURAL"
                    if src_pop == 'E':
                        curr_init = exp_curr_init
                        w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                    else:
                        curr_init = inh_curr_init
                        w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                    static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {}, {"g": init_var("NormalClipped", w_dist), "d": init_var("NormalClippedDelay", d_dist)})
                    srcPop = neuron_populations[src_area][src_pop + layer_map[src_layer]]
                    tarPop = neuron_populations[tar_area][tar_pop + layer_map[tar_layer]]
                    syn_pop = model.add_synapse_population(srcName + "_to_" + tarName if False else "auto", matrix_type, srcPop, tarPop, static_synapse_init, curr_init, init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                    syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / model.dt))
                    if matrix_type == "PROCEDURAL":
                        syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE

    # build & load
    print(f"[WORKER {worker_rank}] Building model: neurons={total_neurons}, synapses={total_synapses}, on device {local_gpu_id}")
    model.build()
    model.load(num_recording_timesteps=buffer_size)

    # simulation loop
    max_steps = duration_timesteps // buffer_size
    step = 0
    all_local_spike_history = []  # optional: 记录本 worker 全历史
    try:
        while step < max_steps:
            step += 1
            # simulate buffer_size steps (buffer_size may be 1)
            for _ in range(buffer_size):
                model.step_time()

            # pull buffers & record spikes
            model.pull_recording_buffers_from_device()
            spike_data_temp = { area: {pop: [] for pop in neuron_populations[area].keys()} for area in neuron_populations.keys() }
            record_spike(neuron_populations, spike_data_temp)
            all_local_spike_history.append(spike_data_temp)  # optional local history

            # send data to master with tag = step
            send_msg = {
                "worker_rank": worker_rank,
                "spike_data": spike_data_temp,
                "NeuronNumber": NeuronNumber_local,
                "timestamp_sent": perf_counter()
            }
            comm.send(send_msg, dest=0, tag=step)

            # wait for master update (tag = step)
            control_msg = comm.recv(source=0, tag=step)
            recv_master_time = perf_counter()
            # immediately send ack (tag = step + max_steps + 1)
            ack_msg = {"worker_rank": worker_rank, "worker_recv_time": recv_master_time}
            comm.send(ack_msg, dest=0, tag=step+max_steps+1)

            if control_msg.get("type") == "stop":
                print(f"[WORKER {worker_rank}] received STOP at step {step}")
                break
            # else control_msg['updates'] is available, apply if needed (TODO)
            updates = control_msg.get("updates", None)
            if updates:
                # 示例访问 updates:
                # for area, pop_dict in updates['rate'].items(): ...
                pass

        # worker finished main loop -> send final ack for stop
        final_ack = {"worker_rank": worker_rank}
        comm.send(final_ack, dest=0, tag=max_steps+2)
    except Exception as e:
        print(f"[WORKER {worker_rank}] exception: {e}")
        raise

    print(f"[WORKER {worker_rank}] exiting cleanly.")
