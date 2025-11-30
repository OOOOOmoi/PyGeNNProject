import os
import sys
import time
import pickle
from config import expLIF_dict, input, layer_map, vis_content, \
    get_NN, get_SN, get_weight, get_weight_ext, externalRates, get_cc_delay, \
    getModelName, remove_dash_from_index_columns, get_ext_rate, net
from visual import visualize
from record import record_spike
import numpy as np
from argparse import ArgumentParser, Namespace
import string
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from pygenn.cuda_backend import DeviceSelect
from time import perf_counter
from itertools import product
import pandas as pd
from collections import defaultdict
from nested_dict import nested_dict
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Process, Queue, Pipe
from collections import defaultdict
from numba import njit, prange, cuda, int32, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import cupy as cp
import math
import multiprocessing as mp
NUM_THREADS_PER_SPIKE = 1
duration = 200
DT_MS = 0.1
duration_timesteps = int(round(duration / DT_MS))
ten_percent_timestep = duration_timesteps // 10
buffer_size = 1

def split_indices(num_areas, num_workkers):
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_workkers - 1) // num_workkers  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workkers) if indices[i*chunk_size:(i+1)*chunk_size]]

def build_spike_buffer(area_num, NN, SN, delay_cc, weight, dt, tar_area_list, net):
    buffer = {}
    weight_array = []
    spike_count = {}
    prob_array = []
    src_pop_num_array = []
    tar_neu_num_array = []
    R_array = []
    all_area = net["area_list"]
    all_area = [s.replace("-", "") for s in all_area]
    all_area_list = [all_area[1], all_area[2], all_area[4], all_area[12]]
    layer_list = net["layer_list"]
    pop_list = net["population_list"]
    for tar_area in tar_area_list:
        for tar_layer in layer_list:
            for tar_pop in pop_list:
                tar = (tar_area, tar_layer, tar_pop)
                n_neu = int(NN.loc[tar])
                Rm = 45.4 if tar_pop == "E" else 100
                src_pop_num = 0
                for src_area in all_area[0:area_num]:
                    for src_layer in layer_list:
                        for src_pop in pop_list:
                            src = (src_area, src_layer, src_pop)
                            spike_count[(src_area, src_pop+layer_map[src_layer])] = 10
                            conn_num = SN.loc[tar, src]
                            w = weight.loc[tar, src] / 1000
                            if conn_num == 0 or NN.loc[tar] == 0 or NN.loc[src] == 0 or src_area == tar_area:
                                continue  # 无连接则跳过
                            prob = conn_num / NN.loc[src] / NN.loc[tar]
                            prob_array.append(prob)
                            # 延迟步长
                            delay_ms = delay_cc.loc[(src_area, tar_area)]
                            delay_step = int(np.ceil(delay_ms / dt))
                            # 初始化 buffer
                            buffer[((tar_area, tar_pop+layer_map[tar_layer]), ((src_area, src_pop+layer_map[src_layer])))] = np.zeros(delay_step, dtype=np.float32)
                            src_pop_num += 1
                            weight_array.append(w)
                if src_pop_num:
                    src_pop_num_array.append(src_pop_num)
                    tar_neu_num_array.append(n_neu)
                    R_array.extend([Rm] * n_neu)
        # convert collected lists to numpy arrays for efficient numeric ops
        weight_array = np.array(weight_array, dtype=np.float32)
        prob_array = np.array(prob_array, dtype=np.float32)
        src_pop_num_array = np.array(src_pop_num_array, dtype=np.int32)
        tar_neu_num_array = np.array(tar_neu_num_array, dtype=np.int32)
        R_array = np.array(R_array, dtype=np.float32)
    return buffer, weight_array, prob_array, src_pop_num_array, tar_neu_num_array, R_array

@njit(parallel=True)
def fast_update_inSyn(spike_array, cum_src, tar_neu_num_array, prob_array, weight_array, inSyn_buffer):
    n_src = len(spike_array)
    for sc_idx in prange(n_src):
        sc = spike_array[sc_idx]
        if sc <= 0:
            continue
        group_idx = np.searchsorted(cum_src, sc_idx + 1)
        start_id = np.sum(tar_neu_num_array[:group_idx]) if group_idx > 0 else 0
        group_neu_count = tar_neu_num_array[group_idx]
        prob = prob_array[sc_idx]
        weight = weight_array[sc_idx]

        # 预估总命中数 = sc * group_neu_count * prob
        total_hits = int(sc * group_neu_count * prob)

        if total_hits > 0:
            # 随机选择目标神经元
            post_idxs = np.random.randint(0, group_neu_count, total_hits)
            for i in range(total_hits):
                inSyn_buffer[start_id + post_idxs[i]] += weight

@cuda.jit
def fast_update_inSyn_gpu(spike_array, cum_src, tar_neu_num_array, prob_array, weight_array, inSyn_buffer, rng_states):
    idx = cuda.grid(1)
    if idx >= spike_array.size:
        return

    sc = int32(spike_array[idx])
    if sc <= 0:
        return

    # binary-search for group index using cum_src (assumed int32 device array)
    left = 0
    right = cum_src.size - 1
    group_idx = 0
    target = idx + 1  # searchsorted semantics used in host code

    while left <= right:
        mid = (left + right) >> 1
        if target <= cum_src[mid]:
            group_idx = mid
            right = mid - 1
        else:
            left = mid + 1

    # compute start_id (sum tar_neu_num_array[:group_idx])
    start_id = int32(0)
    for i in range(group_idx):
        start_id += int32(tar_neu_num_array[i])

    group_neu_count = int32(tar_neu_num_array[group_idx])
    prob = float32(prob_array[idx])
    weight = float32(weight_array[idx])

    total_hits = int32(sc * group_neu_count * prob)
    if total_hits <= 0:
        return

    # draw random posts using rng_states and thread idx
    for _ in range(total_hits):
        # xoroshiro128p_uniform_float32 takes (states, idx) and returns float32 in [0,1)
        u = xoroshiro128p_uniform_float32(rng_states, idx)
        post = int32(u * group_neu_count)
        # atomic add to avoid race conditions
        cuda.atomic.add(inSyn_buffer, start_id + post, weight)

def get_neu_vars_array(neuron_populations, merge=True):
    array_V = []
    array_tref = []
    for area in neuron_populations.keys():
        for pop in neuron_populations[area].keys():
            neu_pop = neuron_populations[area][pop]
            neu_pop.vars["V"].pull_from_device()
            neu_pop.vars["RefracTime"].pull_from_device()
            array_V.append(neu_pop.vars["V"].current_view.copy())
            array_tref.append(neu_pop.vars["RefracTime"].current_view.copy())

    if merge:
        import numpy as np
        array_V = np.concatenate(array_V) if array_V else np.array([])
        array_tref = np.concatenate(array_tref) if array_tref else np.array([])
    return array_V, array_tref

def Part(worker_id, gpu_id,  area_list, NN, rate_ext, SN, weight, delay_cc, weight_ext, area_num,
         to_master: Queue, from_master: Queue, done_queue: Queue):
    print(f"start proccess {worker_id} on GPU {gpu_id}")
    model = GeNNModel("float", f"HMAM_MPI_CODE/worker{worker_id}_on_device{gpu_id}", device_select_method=DeviceSelect.MANUAL, manual_device_id=gpu_id)
    if isinstance(area_list, str):
        area_list = [area_list]
    model.dt = 0.1
    model.fuse_postsynaptic_models = True
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    layer_list = net["layer_list"]
    pop_list = net["population_list"]
    lif_init = {"V": init_var("Uniform", {"max": -50.0, "min": -200.0}), "RefracTime": 0.0}
    poisson_init = {"current": 0.0}
    total_neurons = 0
    NeuronNumber = defaultdict(dict)
    neuron_populations = defaultdict(dict)
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in NN.index:
                    popName = area+pop+layer_map[layer]
                    popNum = NN.loc[(area, layer, pop)]
                    NeuronNumber[area][pop+layer_map[layer]] = popNum
                    if popNum != 0:
                        print("creating neuron group {popName} with {popNum} neurons".format(popName=popName, popNum=popNum))
                        if (pop == "E"):
                            neuronParam = net['neuron_params_E']
                        else:
                            neuronParam = net['neuron_params_I']
                        params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                                    "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                                    "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                                    "TauRefrac": neuronParam['t_ref']}
                        neuron_pop = model.add_neuron_population(popName, popNum, "LIF", params, lif_init)
                        ext_weight = weight_ext.loc[(area, layer, pop)]
                        rate = rate_ext.loc[(area, layer, pop)] * 1
                        # rate = 10*K
                        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                        model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)

                        neuron_pop.spike_recording_enabled = True

                        total_neurons += popNum
                        neuron_populations[area][pop+layer_map[layer]] = neuron_pop

    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 2})
    inh_curr_init = init_postsynaptic("ExpCurr", {"tau": 5})
    total_synapses = 0
    syn_group_num = 0
    for tar_area, src_area in product(area_list, area_list):
        for tar_layer, src_layer in product(layer_list, layer_list):
            for tar_pop, src_pop in product(pop_list, pop_list):
                tar = (tar_area, tar_layer, tar_pop)
                src = (src_area, src_layer, src_pop)
                if tar in SN.index and src in SN.columns:
                    tarName = tar_area+tar_pop+layer_map[tar_layer]
                    srcName = src_area+src_pop+layer_map[src_layer]
                    synName = srcName + "_to_" + tarName
                    synNum = SN.loc[tar, src]
                    wAve = weight.loc[tar, src] / 1000
                    wSd = wAve / 10 / 1000
                    if src_area == tar_area:
                        if src_pop == 'E':
                            meanDelay = net['delay_e']
                            delay_sd = net['delay_e_sd']
                        else:
                            meanDelay = net['delay_i']
                            delay_sd = net['delay_i_sd']
                    else:
                        meanDelay = delay_cc.loc[(src_area, tar_area)]
                        delay_sd = meanDelay / 10
                    if synNum > 0:
                        tarPop = neuron_populations[tar_area][tar_pop+layer_map[tar_layer]]
                        srcPop = neuron_populations[src_area][src_pop+layer_map[src_layer]]
                        quantile = 0.9999
                        normal_quantile_cdf = norm.ppf(quantile)
                        max_delay = meanDelay + (delay_sd * normal_quantile_cdf)
                        connect_params = {"num": synNum}
                        # Build distribution for delay parameters
                        d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_delay}
                        total_synapses += synNum
                        syn_group_num += 1
                        # Build unique synapse name
                        matrix_type = "PROCEDURAL"
                        if src_pop == 'E':
                            curr_init = exp_curr_init
                            w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                        else:
                            curr_init = inh_curr_init
                            w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                        static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                            {"g": init_var("NormalClipped", w_dist),
                                                            "d": init_var("NormalClippedDelay", d_dist)})
                        syn_pop = model.add_synapse_population(synName, matrix_type,
                        srcPop, tarPop,
                        static_synapse_init, curr_init,
                        init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                        
                        syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / model.dt))

                        if matrix_type=="PROCEDURAL":
                            syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
    print(f"Building worker {worker_id} of {total_neurons} neurons and {total_synapses} synapses of {syn_group_num} groups on device {gpu_id}")
    model.build()
    print(f"Loading worker {worker_id} on device {gpu_id}")
    model.load(num_recording_timesteps=buffer_size)
    print(f"Simulating worker {worker_id} on device {gpu_id}")

    # generate spike buffer
    spike_count_buffer, weight_array, prob_array, src_pop_num_array, tar_neu_num_array, R_array  = \
        build_spike_buffer(area_num, NN, SN, delay_cc, weight, dt=model.dt, tar_area_list=area_list, net=net)

    cum_src = np.cumsum(src_pop_num_array)  # cumulative counts
    # inSyn buffer
    inSyn_buffer = np.zeros(len(R_array), dtype=np.float32)
    cuda.select_device(gpu_id)

    # 需要在循环外初始化一次
    d_cum_src   = cuda.to_device(cum_src.astype(np.int32))
    d_tar_num   = cuda.to_device(tar_neu_num_array.astype(np.int32))
    d_prob      = cuda.to_device(prob_array.astype(np.float32))
    d_weight    = cuda.to_device(weight_array.astype(np.float32))
    d_inSyn     = cuda.to_device(inSyn_buffer.astype(np.float32))

    d_spike_array = cuda.device_array(len(spike_count_buffer), dtype=np.int32)

    rng_states = create_xoroshiro128p_states(len(spike_count_buffer), seed=1234)

    threads = 32
    blocks = max(1, math.ceil(len(spike_count_buffer) / threads))

    flag = 0
    current_step = 0
    update_time_V = 0
    update_time_inSyn = 0
    update_time_spike = 0
    update_time = 0
    update_time_decay = 0
    spike_history = []
    while model.t < duration:

        if spike_count_buffer:
            time_start_total = perf_counter()
            time_start = perf_counter()
            # _, array_tref = get_neu_vars_array(neuron_populations)
            spike_array = np.array([ buf[(current_step - len(buf)) % len(buf)] 
                            for buf in spike_count_buffer.values() ], dtype=np.int32)
            time_end = perf_counter()
            update_time_spike = time_end - time_start
            # jit即时编译加速
            time_start = perf_counter()
            fast_update_inSyn(spike_array, cum_src, tar_neu_num_array, prob_array, weight_array, inSyn_buffer)
            time_end = perf_counter()
            update_time_inSyn = time_end - time_start

            # numba.cuda加速
            # time_start = perf_counter()
            # d_spike_array.copy_to_device(spike_array)
            # fast_update_inSyn_gpu[blocks, threads](
            #     d_spike_array,
            #     d_cum_src,
            #     d_tar_num,
            #     d_prob,
            #     d_weight,
            #     d_inSyn,
            #     rng_states
            # )
            # cuda.synchronize()
            # d_inSyn.copy_to_host(inSyn_buffer)
            # time_end = perf_counter()
            # update_time_inSyn = time_end - time_start

            # 原始 numpy 实现（慢）
            # for sc_idx, sc in enumerate(spike_array):
            #     if sc == 0:
            #         continue
            #     group_idx = int(np.searchsorted(cum_src, sc_idx + 1))
            #     start_id = int(np.sum(tar_neu_num_array[:group_idx])) if group_idx > 0 else 0
            #     group_neu_count = int(tar_neu_num_array[group_idx]) if group_idx < len(tar_neu_num_array) else 0
            #     prob = prob_array[sc_idx]
            #     while sc >= 0:
            #         # 按二项分布从 group_neu_count 个目标神经元中抽取命中数 k，再随机选择 k 个目标神经元并累加权重
            #         k = np.random.binomial(group_neu_count, prob)
            #         if k > 0:
            #             if k >= group_neu_count:
            #                 post_idxs = np.arange(start_id, start_id + group_neu_count, dtype=int)
            #             else:
            #                 choices = np.random.choice(group_neu_count, size=k, replace=False)
            #                 post_idxs = start_id + choices.astype(int)
            #             inSyn_buffer[post_idxs] += weight_array[sc_idx]
            #         sc -= 1
            time_start = perf_counter()
            inSyn_buffer *= np.exp(model.dt / 4)
            dv = inSyn_buffer * R_array
            dv = np.asarray(dv)
            inSyn_buffer *= np.exp(model.dt / 2)
            time_end = perf_counter()
            update_time_decay = time_end - time_start

            time_start = perf_counter()
            for area in neuron_populations.keys():
                for pop in neuron_populations[area].keys():
                    # 如果当前 (area, pop) 是某些目标键的 tar，则把本步的 spike_count 累加到对应的 buffer 槽中
                    tar_key = (area, pop)
                    if any(tar_key == k[0] for k in spike_count_buffer.keys()):
                        neu_pop = neuron_populations[area][pop]
                        neu_pop.vars["V"].pull_from_device()
                        neu_pop.vars["RefracTime"].pull_from_device()
                        pop_size = neu_pop.num_neurons
                        array_V = neu_pop.vars["V"].current_view.copy()
                        array_tref = neu_pop.vars["RefracTime"].current_view.copy()
                        dv_ = dv[:pop_size]
                        dv_[array_tref <= 0] = 0.0
                        array_V += dv_
                        neu_pop.vars["V"].current_view[:] = array_V
                        neu_pop.vars["V"].push_to_device()
                        dv = dv[pop_size:]
            time_end = perf_counter()
            update_time_V = time_end - time_start
            time_end_total = perf_counter()
            update_time = time_end_total - time_start_total
        t_start = perf_counter()
        model.step_time()
        t_end = perf_counter()
        step_time = t_end - t_start  # 单次 step_time 耗时（秒）

        if not model.timestep % buffer_size:
            spike_data_temp = {
                area: {pop: [] for pop in neuron_populations[area].keys()}
                for area in neuron_populations.keys()
            }
            model.pull_recording_buffers_from_device()
            record_spike(neuron_populations, spike_data_temp)
            spike_history.append(spike_data_temp)
            # 统计 (只统计，不传spike_raw)
            local_stat = defaultdict(lambda: defaultdict(int))
            for area in neuron_populations.keys():
                for pop in neuron_populations[area].keys():
                    local_stat[area][pop] += len(spike_data_temp[area][pop])   # 当前step群体spike数

            # 把当前进程的数据发给主进程
            msg = {
                "worker_id": worker_id,
                "spike_data": spike_data_temp,
                "timestamp": time.perf_counter(),
                "step_time": step_time,
                "update_V_time": update_time_V,
                "update_inSyn_time": update_time_inSyn,
                "update_spike_time": update_time_spike,
                "update_decay_time": update_time_decay,
                "update_time": update_time
            }

            to_master.put(msg)
            done_queue.put(worker_id)
            # 等待主进程的新指令（同步点）
            msg = from_master.get()
            if msg["type"] == "continue":
                # 可以更新模型参数，例如 rate 或 weight
                updates = msg.get("updates", None)
                if updates:
                    # TODO: 根据 updates 修改模型参数
                    rate = updates["rate"]
                    count_info = updates["spike_count"]
                    if spike_count_buffer:
                        for (tar, src), buf in spike_count_buffer.items():
                            src_area, src_pop = src
                            spike_count = count_info[src_area][src_pop]  # 当前时间步源群体的 spike 数
                            buf[current_step % len(buf)] += spike_count
                        pass
                    pass
            elif msg["type"] == "stop":
                break
        current_step += 1
        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print(f"{worker_id}-th Proccess {flag * 10}%")

    # worker进程写磁盘更好
    with open(f"spike_worker_{worker_id}.pkl", "wb") as f:
        pickle.dump(spike_history, f)
    to_master.put({"worker_id": worker_id, "final_file": f"spike_worker_{worker_id}.pkl"})


def merge_spike_data(spike_data_blocks):
    merged = {}

    for block in spike_data_blocks:
        if block is None:
            continue
        for area, pop_dict in block.items():
            if area not in merged:
                merged[area] = {}
            for pop, spikes in pop_dict.items():
                if pop not in merged[area]:
                    merged[area][pop] = []
                merged[area][pop].extend(spikes)   # 把多个 worker 的数据拼接到一起
    return merged

def split_spike_data_by_area(spike_data):
    return [{area: pop_dict} for area, pop_dict in spike_data.items()]

if __name__ == '__main__':
    area_list = net["area_list"]
    area_list = [s.replace("-", "") for s in area_list]
    layer_list = net["layer_list"]
    pop_list = net["population_list"]

    NN=get_NN()
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
    idx = pd.IndexSlice

    NeuronNumber = defaultdict(dict)
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in NN.index:
                    popNum = NN.loc[(area, layer, pop)]
                    NeuronNumber[area][pop+layer_map[layer]] = popNum

    num_gpus = 2
    procs_per_gpu = 2   # 假设每个GPU跑2个进程
    num_workers = num_gpus * procs_per_gpu
    split_idx = split_indices(4,num_workers)
    # split_idx = [[2], [3], [5], [13]]
    to_master_queues = []
    from_master_queues = []
    processes = []
    # 在主进程创建时，新增一个 done_queue 用于异步通知谁完成了
    done_queue = Queue()
    # 创建双向队列
    for i in range(num_workers):
        to_master = Queue()
        from_master = Queue()
        to_master_queues.append(to_master)
        from_master_queues.append(from_master)
        gpu_id = i % num_gpus
        p = Process(target=Part,
                    args=(i,
                          gpu_id,
                          [area_list[j-1] for j in split_idx[i]],
                          NN, rate_ext, SN, weight, delay_cc, weight_ext, num_workers,
                          to_master, from_master, done_queue))
        p.start()
        processes.append(p)

    # 主循环
    step = 0
    max_steps = duration_timesteps // buffer_size
    all_steps_spike_data = []
    while step < max_steps:
        spike_data_blocks = []

        # ---- 等待所有子进程提交数据 ----
        for i in range(num_workers):
            wid = done_queue.get()       # 谁先完成谁的 id 先到
            msg = to_master_queues[wid].get()
            recv_time = time.perf_counter()

            latency = recv_time - msg["timestamp"]
            data_size = len(pickle.dumps(msg))
            speed_MBps = data_size / (latency * 1024 * 1024)

            print(f"[Round {step}] Worker {msg['worker_id']} -> 主进程: "
                f"延迟 {latency*1000:.3f} ms, "
                f"速度 {speed_MBps:.2f} MB/s, "
                f"大小 {data_size/1024:.1f} KB, "
                f"模拟时间 {msg['step_time']*1000:.3f} ms, "
                f"更新spike array时间 {msg['update_spike_time']*1000:.3f} ms, "
                f"更新电流buffer时间 {msg['update_inSyn_time']*1000:.3f} ms, "
                f"电流buffer衰减时间 {msg['update_decay_time']*1000:.3f} ms, "
                f"更新膜电位时间 {msg['update_V_time']*1000:.3f} ms, "
                f"更新时间 {msg['update_time']*1000:.3f} ms")

            spike_data_blocks.append(msg["spike_data"])

        # ---- 合并 & 统计 ----
        all_spike_data = merge_spike_data(spike_data_blocks)
        all_steps_spike_data.append(all_spike_data)
        processed_data = {"rate": {}, "spike_count": {}}
        for area, pop_dict in all_spike_data.items():
            processed_data["rate"][area] = {}
            processed_data["spike_count"][area] = {}
            for pop, data_chunks in pop_dict.items():
                if not data_chunks:
                    processed_data["rate"][area][pop] = 0.0
                    processed_data["spike_count"][area][pop] = 0
                    continue
                all_spikes = np.vstack(data_chunks)
                spike_count = all_spikes.shape[0]
                num_neurons = NeuronNumber[area][pop]
                spike_rate = spike_count / num_neurons * 1000
                processed_data["rate"][area][pop] = spike_rate
                processed_data["spike_count"][area][pop] = spike_count

        # ---- 广播给子进程继续跑 ----
        for q in from_master_queues:
            q.put({"type": "continue", "updates": processed_data})

        step += 1

    # ---- 仿真完成，发送 stop ----
    for q in from_master_queues:
        q.put({"type": "stop"})

    # ---- 等待所有子进程退出 ----
    for p in processes:
        p.join()

    print("所有子进程已结束，主进程退出。")
    final_spike_data = merge_spike_data(all_steps_spike_data)
    for area, area_dict in final_spike_data.items():
        spike_data_temp = {}
        spike_data_temp[area] = area_dict
        visualize(suffix="test", spike_data=spike_data_temp, duration=1000,
                model_name="HMAM", NeuronNumber=NeuronNumber, drop=0)