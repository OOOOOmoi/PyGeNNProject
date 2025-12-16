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
import time
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from nested_dict import nested_dict
from config import collection_params, vis_content, record_I
from getStruct import getWeightMap, getDelayMap, get_struct, has_key_path, getWeightMap_full_type
from visual import visualize, generate_unique_suffix
from connectom import connectom
from record import record_spike, save_spike, record_inSyn, save_inSyn
from expLIF import expLIF_model
from multiprocessing import Process, Queue, Pipe
from numba import njit, prange, cuda, int32, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
DT_MS=0.1
NUM_THREADS_PER_SPIKE = 1
MAX_SHARED_BINS = 1024
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
buffer_size = 1
duration = 200
duration_timesteps = duration / DT_MS
stim_start = 400
stim_end = 800
def split_indices(num_areas, num_workkers):
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_workkers - 1) // num_workkers  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workkers) if indices[i*chunk_size:(i+1)*chunk_size]]


def build_spike_buffer(area_num, NN, SN, delay_cc, weight, dt, tar_area_list, all_area, pop_list, gL):
    buffer = {}
    weight_array = []
    prob_array = []
    src_pop_num_array = []
    tar_neu_num_array = []
    R_array = []
    for tar_area in tar_area_list:
        for tar_pop in pop_list:
            n_neu = int(NN[tar_area][tar_pop])
            Rm = 1 / gL[tar_pop] * 1000
            src_pop_num = 0
            for src_area in all_area[0:area_num]:
                for src_pop in pop_list:
                    conn_num = SN[tar_area][tar_pop][src_area][src_pop]
                    w = weight[tar_area][tar_pop][src_area][src_pop] / 1000 * 0
                    if conn_num == 0 or NN[tar_area][tar_pop] == 0 or NN[src_area][src_pop] == 0 or src_area == tar_area:
                        continue  # 无连接则跳过
                    prob = conn_num / NN[src_area][src_pop] / NN[tar_area][tar_pop]
                    prob_array.append(prob)
                    # 延迟步长
                    delay_ms = delay_cc[tar_area][tar_pop][src_area][src_pop]['ave']
                    delay_step = int(np.ceil(delay_ms / dt))
                    # 初始化 buffer
                    buffer[((tar_area, tar_pop), ((src_area, src_pop)))] = np.zeros(delay_step, dtype=np.float32)
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


@cuda.jit
def fast_update_inSyn_gpu(
    spike_array,           # int32[n_src]
    cum_src,               # int32[n_group]
    tar_neu_num_array,     # int32[n_group]
    prob_array,            # float32[n_src]
    weight_array,          # float32[n_src]
    inSyn_buffer,          # float32[sum_tar_neu]
    rng_states             # xoroshiro states
):
    # blockIdx.x 对应 sc_idx（一个 block 处理一个 sc）
    sc_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if sc_idx >= spike_array.size:
        return

    sc = int32(spike_array[sc_idx])
    if sc <= 0:
        return

    # binary search (searchsorted behavior)
    target = sc_idx + 1
    left = 0
    right = cum_src.size - 1
    group_idx = 0
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
    prob = float32(prob_array[sc_idx])
    weight = float32(weight_array[sc_idx])

    if group_neu_count <= 0:
        return
    
    n_trials = sc * group_neu_count

    if n_trials < 2048:
        # 用 严谨 Bernoulli 累加（方案1）
        # 每线程分摊 trials
        trials_per_thread = (n_trials + n_threads - 1) // n_threads
        base = tid * trials_per_thread

        local_hits = int32(0)
        rng_base = sc_idx * n_threads

        for i in range(trials_per_thread):
            t = base + i
            if t < n_trials:
                u = xoroshiro128p_uniform_float32(
                    rng_states,
                    rng_base + (t % n_threads)
                )
                if u < prob:
                    local_hits += 1

        # block 归约
        shared_hits = cuda.shared.array(shape=1024, dtype=int32)  # threads <=1024
        shared_hits[tid] = local_hits
        cuda.syncthreads()

        # 并行 reduction
        stride = n_threads // 2
        while stride > 0:
            if tid < stride:
                shared_hits[tid] += shared_hits[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            total_hits = shared_hits[0]
    else:
        # 用 Poisson 近似（方案2）
        if gid == 0:
            lam = float32(sc * group_neu_count) * prob

            # Knuth 算法（GPU 版 Poisson）
            L = math.exp(-lam)
            k = int32(0)
            p_acc = float32(1.0)

            rng_base = sc_idx * n_threads + tid

            while p_acc > L:
                k += 1
                u = xoroshiro128p_uniform_float32(rng_states, rng_base)
                p_acc *= u

            total_hits = k - 1
    if total_hits <= 0:
        return

    # 如果 group_neu_count 很小，使用 shared local histogram，最后一次性写回全局
    if group_neu_count <= MAX_SHARED_BINS:
        # 定义 shared bins（静态大小）
        shared_bins = cuda.shared.array(shape=MAX_SHARED_BINS, dtype=float32)
        # 每个线程清零分块（因为 MAX_SHARED_BINS 可能大于 blockDim）
        # 每个线程负责清理若干个槽
        stride = n_threads
        for i in range(tid, group_neu_count, stride):
            shared_bins[i] = 0.0
        cuda.syncthreads()

        # 为了分散 RNG 消耗，每个线程生成 hits_per_thread 个 hits
        hits_per_thread = (total_hits + n_threads - 1) // n_threads
        base = tid * hits_per_thread
        rng_base = sc_idx * n_threads  # 区分每个 block 的 rng 段
        for i in range(hits_per_thread):
            h = base + i
            if h < total_hits:
                # rng index use (rng_base + tid) to avoid identical sequence across threads
                u = xoroshiro128p_uniform_float32(rng_states, rng_base + (i % n_threads))
                post = int32(u * group_neu_count)
                # 增量写 shared
                # 这里使用 shared 的 atomic（在 CUDA 中是快速的）
                cuda.atomic.add(shared_bins, post, weight)

        cuda.syncthreads()
        # block 内一个线程把 shared_bins 写回全局（按段写回以降低全局冲突）
        # 我们分散写回：每个线程写回一段槽
        for i in range(tid, group_neu_count, stride):
            val = shared_bins[i]
            if val != 0.0:
                cuda.atomic.add(inSyn_buffer, start_id + i, val)

    else:
        # group too large -> every thread generates hits and atomic to global
        hits_per_thread = (total_hits + n_threads - 1) // n_threads
        base = tid * hits_per_thread
        rng_base = sc_idx * n_threads
        for i in range(hits_per_thread):
            h = base + i
            if h < total_hits:
                u = xoroshiro128p_uniform_float32(rng_states, rng_base + (i % n_threads))
                post = int32(u * group_neu_count)
                cuda.atomic.add(inSyn_buffer, start_id + post, weight)

@cuda.jit
def decay_gpu(inSyn, R, IR, dt):
    i = cuda.grid(1)
    if i < inSyn.size:
        inSyn[i] *= math.exp(-dt / 4)
        IR[i] = inSyn[i] * R[i]
        inSyn[i] *= math.exp(-dt / 2)

def get_neu_vars_array(neuron_populations, spike_count_buffer, merge=True):
    array_V = []
    array_tref = []
    for area in neuron_populations.keys():
        for pop in neuron_populations[area].keys():
            tar_key = (area, pop)
            if any(tar_key == k[0] for k in spike_count_buffer.keys()):
                neu_pop = neuron_populations[area][pop]
                neu_pop.vars["V"].pull_from_device()
                neu_pop.vars["RefracTime"].pull_from_device()
                array_V.append(neu_pop.vars["V"].current_view.copy())
                array_tref.append(neu_pop.vars["RefracTime"].current_view.copy())
    if merge:
        array_V = np.concatenate(array_V) if array_V else np.array([])
        array_tref = np.concatenate(array_tref) if array_tref else np.array([])
    return array_V, array_tref


def prepare():
    DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
    with open(DataPath, 'r') as f:
        ParamOfAll = json.load(f)
    SynapsesNumber=ParamOfAll["synapses"]
    NeuronNumber=ParamOfAll["neuron_numbers"]
    Dist=ParamOfAll["distances"]
    area_list=ParamOfAll["area_list"]
    pop_list=ParamOfAll["population_list"]
    model_structure = get_struct()
    # SynapsesWeightMean, SynapsesWeightSd = getWeightMap(model_structure, args)
    SynapsesWeightMean, SynapsesWeightSd = getWeightMap_full_type(model_structure)
    delayMap = getDelayMap(model_structure, Dist)
    return NeuronNumber, SynapsesNumber, SynapsesWeightMean, SynapsesWeightSd, delayMap, area_list, pop_list



def Part(worker_id, gpu_id,  area_list, all_area, pop_list, NN, SN, weight, delay_cc, area_num, 
         to_master: Queue, from_master: Queue, done_queue: Queue, final_queue: Queue):
    print(f"start proccess {worker_id} on GPU {gpu_id}")
    model = GeNNModel("float", f"GenCODE/worker{worker_id}_on_device{gpu_id}", device_select_method=DeviceSelect.MANUAL, manual_device_id=gpu_id)
    if isinstance(area_list, str):
        area_list = [area_list]
    model.dt = 0.1
    model.fuse_postsynaptic_models = True
    model.default_narrow_sparse_ind_enabled = True
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
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
    neuronParam=collection_params['single_neuron_dict']
    params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                "Vthresh" : neuronParam['V_th'], "Ioffset": 0,
                "TauRefrac": neuronParam['t_ref']}
    exc_exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})
    inh_exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": params['TauRefrac']}
    input=collection_params['connection_params']['input']
    stim_info=collection_params['stim']
    # print("Creating neuron populations:")
    total_neurons = 0
    neuron_populations = defaultdict(dict)
    poisson_init = {"current": 0.0}
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": params['TauRefrac']}
    Cm = collection_params['single_neuron_dict']['Cm']
    gL = collection_params['single_neuron_dict']['gL']
    tref = collection_params['single_neuron_dict']['tref']
    Vrest = collection_params['single_neuron_dict']['Vrest']
    Vth = collection_params['single_neuron_dict']['Vth']
    rate_ext = collection_params['single_neuron_dict']['rate_ext']
    for area in area_list:
        for pop in pop_list:
            popName = area+pop
            params["C"] = Cm[pop] / 1000.0
            params["TauM"] = Cm[pop] / gL[pop]
            params["Vrest"] = Vrest[pop]
            params["Vreset"] = Vrest[pop] - 10.0
            params["Vthresh"] = Vth[pop]
            params["TauRefrac"] = tref[pop]
            pop_size = NeuronNumber[area][pop]
            if pop_size > 0:
                neuron_pop = model.add_neuron_population(popName, pop_size, "LIF", params, lif_init)
                if has_key_path(stim_info, area, pop):
                    s=stim_info[area][pop]
                    model.add_current_source(area + pop + '_pulse',
                        trigger_pulse_model, neuron_pop,
                        {   "start_time":stim_start,
                            "end_time":stim_end,
                            "magnitude": s/1000.0},
                )

                ext_weight = weight[area][pop]['external']['external']
                rate = SN[area][pop]['external']['external'] / NN[area][pop] / 1000
                # rate = rate_ext[pop]/1000
                poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                model.add_current_source(area + pop + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)
                # Enable spike recording
                neuron_pop.spike_recording_enabled = True

                # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
                total_neurons += pop_size
                neuron_populations[area][pop] = neuron_pop
    total_synapses = 0
    synapse_populations = nested_dict()
    for areaTar, areaSrc in product(area_list, area_list):
        for popTar, popSrc in product(pop_list, pop_list):
            wAve = weight[areaTar][popTar][areaSrc][popSrc]/1000.0
            wSd = weight[areaTar][popTar][areaSrc][popSrc]/1000.0/10
            synNum = SN[areaTar][popTar][areaSrc][popSrc]
            tarName = areaTar+popTar
            srcName = areaSrc+popSrc
            synName = srcName+"2"+tarName
            meanDelay=delayMap[areaTar][popTar][areaSrc][popSrc]['ave']
            delay_sd=delayMap[areaTar][popTar][areaSrc][popSrc]['sd']
            max_d=delayMap[areaTar][popTar][areaSrc][popSrc]['max']
            if(synNum>0):
                connect_params = {"num": synNum}
                # Build distribution for delay parameters
                d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0.0, "max": max_d}
                total_synapses += synNum
                # Build unique synapse name
                matrix_type = "PROCEDURAL"
                if popSrc.startswith("E"):
                    w_dist = {"mean": wAve, "sd": wSd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                else:
                    w_dist = {"mean": wAve, "sd": wSd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                
                static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                    {"g": init_var("NormalClipped", w_dist),
                                                    "d": init_var("NormalClippedDelay", d_dist)})
                if popSrc[0] == 'E':
                    syn_pop = model.add_synapse_population(synName, matrix_type,
                                neuron_populations[areaSrc][popSrc], neuron_populations[areaTar][popTar],
                                static_synapse_init, exc_exp_curr_init,
                                init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                else:
                    syn_pop = model.add_synapse_population(synName, matrix_type,
                                neuron_populations[areaSrc][popSrc], neuron_populations[areaTar][popTar],
                                static_synapse_init, inh_exp_curr_init,
                                init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                # Set max dendritic delay and span type
                syn_pop.max_dendritic_delay_timesteps = int(round(max_d / DT_MS))
                if matrix_type=="PROCEDURAL":
                    syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE
                synapse_populations[areaTar][popTar][areaSrc][popSrc] = syn_pop
            else:
                synapse_populations[areaTar][popTar][areaSrc][popSrc] = None
        print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))
    
    print(f"Building worker {worker_id} of {total_neurons} neurons and {total_synapses} synapses on device {gpu_id}")
    model.build()
    print(f"Loading worker {worker_id} on device {gpu_id}")
    model.load(num_recording_timesteps=buffer_size)
    print(f"Simulating worker {worker_id} on device {gpu_id}")

    print("Simulating")
    # generate spike buffer
    spike_count_buffer, weight_array, prob_array, src_pop_num_array, tar_neu_num_array, R_array  = \
        build_spike_buffer(area_num, NN, SN, delay_cc, weight, dt=model.dt, tar_area_list=area_list,
                           all_area=all_area, pop_list=pop_list, gL=gL)

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
    d_R         = cuda.to_device(R_array.astype(np.float32))
    d_IR = cuda.device_array_like(d_inSyn)  # 用作中间 IR 存储
    N_inSyn = len(inSyn_buffer)  # 先前已知
    IR = np.empty_like(inSyn_buffer, dtype=np.float32)
    threads_per_block = 128
    blocks_per_grid = (N_inSyn + threads_per_block - 1) // threads_per_block

    # 获得 GPU 上的 spike_array 存储，以及更新inSyn的 kernel 配置
    d_spike_array = cuda.device_array(len(spike_count_buffer), dtype=np.int32)
    threads = 128  # 或 256，视 GPU 而定；128 是比较保守的起点
    blocks = len(spike_count_buffer)  # 一个 block per sc
    rng_states = create_xoroshiro128p_states(blocks * threads, seed=1234)

    current_step = 0
    local_spike_history = []  # 每步 append 完整 spike_data_temp，用于仿真结束后一次性发回主进程
    buffer_keys = list(spike_count_buffer.keys())  # 固定顺序（非常重要）
    tar_key_set = set(k[0] for k in spike_count_buffer.keys())

    pop_offsets = {}
    offset = 0
    for area in sorted(neuron_populations.keys()):   # 保持确定性顺序
        for pop in sorted(neuron_populations[area].keys()):
            if (area, pop) in tar_key_set:
                size = neuron_populations[area][pop].num_neurons
                pop_offsets[(area, pop)] = (offset, size)
                offset += size


    while model.t < duration:
        if spike_count_buffer:
            t0 = perf_counter()
            # 1) 读取：取 "当前步后 delay 步到达" 的槽
            #    read_idx = (current_step - 1) % L  (t=0 -> last slot)
            spike_list = []
            read_indices = []   # 保存对应的 read_idx，后面写回时复用
            for k in buffer_keys:
                buf = spike_count_buffer[k]
                L = len(buf)
                read_idx = (current_step - 1) % L
                v = int(buf[read_idx])
                # 立即清零，防止累积 / 下次重复读取
                buf[read_idx] = 0
                spike_list.append(v)
                read_indices.append((k, read_idx))  # 记录以便后面写回同一槽
            spike_array = np.array([ buf[current_step % len(buf)] 
                            for buf in spike_count_buffer.values() ], dtype=np.int32)
            t2 = perf_counter()
            # 2) 更新 inSyn（numba jitted）
            # t3 = perf_counter()
            # fast_update_inSyn(spike_array, cum_src, tar_neu_num_array, prob_array, weight_array, inSyn_buffer)
            # t4 = perf_counter()

            # 2) 更新 inSyn (numba.cuda加速)
            t3 = perf_counter()
            d_spike_array.copy_to_device(spike_array)
            fast_update_inSyn_gpu[blocks, threads](
                d_spike_array,
                d_cum_src,
                d_tar_num,
                d_prob,
                d_weight,
                d_inSyn,
                rng_states
            )
            cuda.synchronize()
            t4 = perf_counter()

            # 3) 拉取神经元变量并更新膜电位（尽量减少拷贝）
            t10 = perf_counter()
            # array_V, _ = get_neu_vars_array(neuron_populations, spike_count_buffer)
            model.step_time()
            t11 = perf_counter()

            # 4) 衰减和计算 IR
            t5 = perf_counter()
            decay_gpu[blocks_per_grid, threads_per_block](d_inSyn, d_R, d_IR, model.dt)
            cuda.synchronize()
            d_IR.copy_to_host(IR)
            t6 = perf_counter()

            # 5) 将 IR 应用到各 population（注意 slice 的 offset 必须与 build_spike_buffer 中的 group 划分一致）
            offset = 0
            for area in neuron_populations.keys():
                for pop in neuron_populations[area].keys():
                    # 如果当前 (area, pop) 是某些目标键的 tar，则把本步的 spike_count 累加到对应的 buffer 槽中
                    if (area, pop) in tar_key_set:
                        neu_pop = neuron_populations[area][pop]
                        neu_pop.vars["V"].pull_from_device()
                        neu_pop.vars["RefracTime"].pull_from_device()
                        pop_size = neu_pop.num_neurons
                        array_V_tmp = neu_pop.vars["V"].current_view.copy()
                        array_tref_tmp = neu_pop.vars["RefracTime"].current_view.copy()
                        dv = IR[offset : offset + pop_size] * model.dt / neu_pop.params["TauM"].value
                        dv[array_tref_tmp <= 0] = 0.0
                        array_V_tmp += dv
                        neu_pop.vars["V"].current_view[:] = array_V_tmp
                        neu_pop.vars["V"].push_to_device()
                        offset += pop_size
            t9 = perf_counter()
            # 子进程本地打印自己想要的时间统计（主进程不再收到这些字段）
            print(f"[Worker {worker_id}] timestep={model.timestep} step_time={(t11-t10)*1000:.3f} ms "
                  f"update_V={(t9-t6)*1000:.3f} ms update_inSyn={(t4-t3)*1000:.3f} ms "
                  f"update_decay={(t6-t5)*1000:.3f} ms update_spike={(t2 -t0)*1000:.3f} ms "
                  f"total_update={(t9-t0)*1000:.3f} ms")
        else:
            t10 = perf_counter()
            model.step_time()
            t11 = perf_counter()
            # 子进程本地打印自己想要的时间统计（主进程不再收到这些字段）
            print(f"[Worker {worker_id}] timestep={model.timestep} step_time={(t11-t10)*1000:.3f} ms ")

        # 6) 记录 spikes -> 本地保存 -> 生成 processed 发给主进程
        spike_data_temp = { area: {pop: [] for pop in neuron_populations[area].keys()} for area in neuron_populations.keys() }
        model.pull_recording_buffers_from_device()
        record_spike(neuron_populations, spike_data_temp)

        # ---- 本地累积完整 spike 数据（不发送） ----
        local_spike_history.append(spike_data_temp)

        # ---- 构建精简的 processed（仅 spike_count），发送给主进程 ----
        processed = {"spike_count": {}}
        for area, pop_dict in spike_data_temp.items():
            processed["spike_count"][area] = {}
            for pop, data_chunks in pop_dict.items():
                if not data_chunks:
                    processed["spike_count"][area][pop] = 0
                    continue
                all_spikes = np.vstack(data_chunks)
                spike_count = all_spikes.shape[0]
                processed["spike_count"][area][pop] = int(spike_count)

        # 把当前进程的数据发给主进程
        # 仅发送精简消息到主进程（主进程用 timestamp 计算 latency）
        msg = {
            "worker_id": worker_id,
            "processed": processed,
            "timestamp": time.perf_counter()
        }
        to_master.put(msg)

        # signal 主进程该 worker 本步完成
        done_queue.put(worker_id)
        # 7) 等主进程回复并写回到 same read slot（表示这些 spikes 会在 t + delay 到达）
        msg = from_master.get()
        if msg["type"] == "continue":
            if spike_count_buffer:
                # 可以更新模型参数，例如 rate 或 weight
                updates = msg.get("updates", None)
                if updates:
                    count_info = updates["spike_count"]
                    for (k, idx) in read_indices:
                        # k 是 (tar, src)
                        (tar, src) = k
                        src_area, src_pop = src
                        # 从主进程聚合数据中取出对应 source 的 spike_count（若没有则 0）
                        spike_count = count_info.get(src_area, {}).get(src_pop, 0)
                        spike_count_buffer[k][idx] = spike_count
        elif msg["type"] == "stop":
            break
        current_step += 1

    # 仿真循环结束：一次性把 local_spike_history 发回主进程用于绘图（可能很大）
    final_msg = {"worker_id": worker_id, "final_spike_data": local_spike_history}
    final_queue.put(final_msg)

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

if __name__ == "__main__":
    num_gpus = 10
    procs_per_gpu = 1
    num_workers = 32
    split_idx = split_indices(num_workers,num_workers)
    NN, SN, weight, _, delayMap, area_list, pop_list = prepare()
    NeuronNumber = defaultdict(dict)
    for area in area_list:
        area_num = 0
        for pop in pop_list:
            if has_key_path(NN, area, pop):
                popNum = NN[area][pop]
                area_num += popNum
                NeuronNumber[area][pop] = popNum
        NeuronNumber[area]['total'] = area_num

    to_master_queues = []
    from_master_queues = []
    processes = []
    # 在主进程创建时，新增一个 done_queue 用于异步通知谁完成了
    done_queue = Queue()
    final_queue = Queue()
    # 创建双向队列
    for i in range(num_workers):
        to_master = Queue()
        from_master = Queue()
        to_master_queues.append(to_master)
        from_master_queues.append(from_master)
        gpu_id = i % num_gpus
        assigned_areas = [area_list[j-1] for j in split_idx[i]]
        p = Process(target=Part,
                    args=(i,
                          gpu_id,
                          assigned_areas, area_list, pop_list,
                          NN, SN, weight, delayMap, num_workers,
                          to_master, from_master, done_queue, final_queue))
        p.start()
        processes.append(p)


    # 主循环
    step = 0
    max_steps = duration_timesteps // buffer_size
    while step < max_steps:
        per_worker_processed = []
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
                f"大小 {data_size/1024:.1f} KB")
            per_worker_processed.append(msg["processed"]["spike_count"])

        # 聚合 spike_count（把所有 worker 的计数相加），并重算 rate
        agg_counts = {}
        for block in per_worker_processed:
            for area, pops in block.items():
                agg_counts.setdefault(area, {})
                for pop, cnt in pops.items():
                    agg_counts[area][pop] = agg_counts[area].get(pop, 0) + int(cnt)

        processed_data = {"spike_count": {}}
        for area, pops in agg_counts.items():
            processed_data["spike_count"][area] = {}
            for pop, cnt in pops.items():
                processed_data["spike_count"][area][pop] = int(cnt)


        # ---- 广播给子进程继续跑 ----
        for q in from_master_queues:
            q.put({"type": "continue", "updates": processed_data})

        step += 1

    # ---- 仿真完成，发送 stop ----
    for q in from_master_queues:
        q.put({"type": "stop"})

     # 等待并收集子进程把 final spike history 放到 final_queue（每个 worker 一个大消息）
    final_blocks = []
    for _ in range(num_workers):
        final_msg = final_queue.get()  # blocks until next final arrives
        print(f"主进程收到 worker {final_msg['worker_id']} 的 final_spike_data ({len(final_msg['final_spike_data'])} timesteps).")
        final_blocks.append(final_msg["final_spike_data"])

    # ---- 等待所有子进程退出 ----
    for p in processes:
        p.join()

    # 合并并绘图（使用你原来的 merge_spike_data + visualize 流程）
    final_spike_data = {}
    for b in final_blocks:
        merged = merge_spike_data(b)  # 注意：merge_spike_data 需要接受 list-of-blocks 风格，这里 b 是列表
        # merge 到 final_spike_data
        for area, pop_dict in merged.items():
            final_spike_data.setdefault(area, {})
            for pop, spikes in pop_dict.items():
                final_spike_data[area].setdefault(pop, [])
                final_spike_data[area][pop].extend(spikes)

    print("所有子进程已结束，主进程退出。")
    for area, area_dict in final_spike_data.items():
        spike_data_temp = {}
        spike_data_temp[area] = area_dict
        save_spike(spike_data_temp)
        visualize("Test", spike_data_temp, duration=duration, drop=0, neurons_per_group=200, 
                group_spacing=20, NeuronNumber=NeuronNumber, vis_content=vis_content)