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
from getStruct import getWeightMap, getDelayMap, get_struct, has_key_path, getWeightMap_full_type, getInd
from visual import visualize, generate_unique_suffix
from connectom import connectom
from record import record_spike, save_spike, record_inSyn, save_inSyn
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
duration = 10000
duration_timesteps = duration / DT_MS
stim_start = 400
stim_end = 800

# ===============================================================
# 进程区域索引生成函数，生成 1 ~ num_areas 的列表，然后均匀划分为 num_workers 份
# ===============================================================
def split_indices(num_areas, num_workkers):
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_workkers - 1) // num_workkers  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workkers) if indices[i*chunk_size:(i+1)*chunk_size]]

# ===============================================================
# 生成缓冲区
# 返回值：
# buffer: dict of ((tar_area, tar_pop), (src_area, src_pop)) -> np.ndarray of shape (delay_steps,)，用于存储延迟突触输入，缓冲区的主体
# weight_array: np.ndarray of shape (n_src,), 存储每个突触群的权重均值，n_src 为所有源突触群的总数
# prob_array: np.ndarray of shape (n_src,), 存储每个突触群的连接概率
# src_pop_num_array: np.ndarray of shape (n_groups,), 存储每个目标群接收的源突触群数量
# tar_neu_num_array: np.ndarray of shape (n_groups,), 存储每个目标群的神经元数量
# R_array: np.ndarray of shape (total_target_neurons,), 存储每个目标神经元的膜阻值 Rm，用于后续的 IR 计算
# 所用参数：
# area_num: 当前进程负责的区域数量
# NN: 神经元数量表
# SN: 突触连接数表
# delay_cc: 区域间延迟表
# weight: 突触权重表
# dt: 仿真时间步长
# tar_area_list: 当前进程负责的区域列表
# net: 网络配置字典
# ===============================================================
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
                    w = weight[tar_area][tar_pop][src_area][src_pop] / 1000
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
        weight_array = np.array(weight_array, dtype=np.float32)
        prob_array = np.array(prob_array, dtype=np.float32)
        src_pop_num_array = np.array(src_pop_num_array, dtype=np.int32)
        tar_neu_num_array = np.array(tar_neu_num_array, dtype=np.int32)
        R_array = np.array(R_array, dtype=np.float32)
    return buffer, weight_array, prob_array, src_pop_num_array, tar_neu_num_array, R_array

# ===============================================================
# 快速更新 inSyn 缓冲区函数（GPU 版，使用 numba cuda）
# 输入参数（注意这里所用参数均在设备端）：
# spike_array: np.ndarray of shape (n_src,), 存储每个源突触群在当前时间步的 spike count
# cum_src: np.ndarray of shape (n_groups,), 存储源突触群数量的累积和，用于定位源突触群所属的目标群体
# tar_neu_num_array: np.ndarray of shape (n_groups,), 存储每个目标群的神经元数量
# prob_array: np.ndarray of shape (n_src,), 存储每个源突触群的连接概率
# weight_array: np.ndarray of shape (n_src,), 存储每个源突触群的权重均值
# inSyn_buffer: np.ndarray of shape (total_target_neurons,), 存储所有目标神经元的 inSyn 缓冲区，将被更新
# rng_states: 预先初始化的 XOROSHIRO128+ 随机数生成器状态数组
# 核函数说明：
# 该核函数每个 block 处理一个源突触群（sc_idx），每个（或者多个）线程负责生成部分随机数并更新 inSyn_buffer
# ===============================================================
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
    # tid 对应线程块内线程索引
    # n_threads 对应线程块大小
    # gid 对应全局线程索引
    sc_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if sc_idx >= spike_array.size:
        return

    sc = int32(spike_array[sc_idx])
    if sc <= 0:
        return

    # 二分查找：找到 group_idx 使得 cum_src[group_idx] >= sc_idx + 1
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

    # 计算start_id
    start_id = int32(0)
    for i in range(group_idx):
        start_id += int32(tar_neu_num_array[i])

    group_neu_count = int32(tar_neu_num_array[group_idx])
    prob = float32(prob_array[sc_idx])
    weight = float32(weight_array[sc_idx])

    if group_neu_count <= 0:
        return
    # 合并sc次binomial，相当于做sc * group_neu_count次二项分布实验，每次命中概率为 prob
    # 在这里有个额外的含义，也就是一个block内所有线程共同完成 sc * group_neu_count 次实验
    n_trials = sc * group_neu_count

    # 这里采用两种路径：
    # 1. 如果 n_trials 较小（<2048），则用 严谨的 Bernoulli 累加（每个线程负责部分实验，最后 block 内归约）
    # 2. 如果 n_trials 较大，则用 Poisson 近似（每个 block 只需一个线程生成总命中数）
    if n_trials < 2048:
        # 用 严谨 Bernoulli 累加（方案1）
        # 每线程分摊 trials
        trials_per_thread = (n_trials + n_threads - 1) // n_threads
        # 计算每线程的起始实验索引
        base = tid * trials_per_thread

        local_hits = int32(0)
        # 计算每个 block 的 rng 段起始索引
        rng_base = sc_idx * n_threads

        # 每个线程负责 trials_per_thread 次实验
        for i in range(trials_per_thread):
            t = base + i
            if t < n_trials:
                u = xoroshiro128p_uniform_float32(
                    rng_states,
                    rng_base + (t % n_threads)
                )
                if u < prob:
                    local_hits += 1

        # block 归约，这里使用共享内存，block之间的共享内存是独立的
        # 先把每个线程的 local_hits 写入 shared
        # threads <=1024，所以可以直接静态分配
        shared_hits = cuda.shared.array(shape=1024, dtype=int32)
        shared_hits[tid] = local_hits
        cuda.syncthreads()

        # 并行规约，一种常见的核函数写法，能够降低线程分化带来的影响
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
        # 只需要一个线程生成 total_hits，目前主流的模拟框架都采用的是这种方案
        if gid == 0:
            # 计算 lambda
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
    # 获取了总命中数量 total_hits 后，开始分配到各个 neuron 上
    # 如果 group_neu_count 很小（<= MAX_SHARED_BINS），使用共享内存加速，最后一次性写回全局
    # 不过一般用不上这个功能，这是为了后续扩展准备的
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
                # 生成随机索引
                u = xoroshiro128p_uniform_float32(rng_states, rng_base + (i % n_threads))
                post = int32(u * group_neu_count)
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
        # 直接全局原子操作版本
        hits_per_thread = (total_hits + n_threads - 1) // n_threads
        base = tid * hits_per_thread
        rng_base = sc_idx * n_threads
        for i in range(hits_per_thread):
            h = base + i
            if h < total_hits:
                u = xoroshiro128p_uniform_float32(rng_states, rng_base + (i % n_threads))
                post = int32(u * group_neu_count)
                cuda.atomic.add(inSyn_buffer, start_id + post, weight)

# ===============================================================
# 衰减 inSyn 并计算 IR 的函数（Numba CUDA 版）
# 输入参数：
# inSyn: numba.cuda.device_array of shape (n_neurons,), 存储每个神经元的 inSyn 缓冲区
# R: numba.cuda.device_array of shape (n_neurons,), 存储每个神经元的膜阻值 Rm
# dt: 时间步长
# 返回值：
# IR: numba.cuda.device_array of shape (n_neurons,), 存储每个神经元的 IR 值，用于后续的膜电位更新
# ===============================================================
@cuda.jit
def decay_gpu(inSyn, R, IR, dt):
    i = cuda.grid(1)
    if i < inSyn.size:
        inSyn[i] *= math.exp(-dt / 4)
        IR[i] = inSyn[i] * R[i]
        inSyn[i] *= math.exp(-dt / 2)

# ===============================================================
# 获取神经元变量数组函数，主要是当前时刻的膜电位和不应期
# 输入参数：
# neuron_populations: dict of area -> population -> neuron_population 对象
# spike_count_buffer: dict of ((tar_area, tar_pop), (src_area, src_pop)) -> np.ndarray of shape (delay_steps,)
# merge: bool, 是否将各群体的数组合并为一个大数组返回
# 返回值：
# array_V: np.ndarray or list of np.ndarray, 存储膜电位 V 的数组
# array_tref: np.ndarray or list of np.ndarray, 存储不应期 RefracTime 的数组
# ===============================================================
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

# ===============================================================
# 准备函数，加载参数文件并生成所需的各种映射
# ===============================================================
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
    Ind = getInd(SynapsesNumber, NeuronNumber)
    return NeuronNumber, SynapsesNumber, SynapsesWeightMean, SynapsesWeightSd, delayMap, area_list, pop_list, Ind

# ===============================================================
# 子进程函数 Part，用于每个工作进程创建和运行神经网络模型
# 输入参数：
# worker_id: 工作进程 ID
# gpu_id: 分配给该进程的 GPU ID
# area_list: 该进程负责的脑区列表
# NN: 神经元数量 DataFrame
# rate_ext: 外部输入噪声强度 DataFrame
# SN: 突触数量 DataFrame
# weight: 突触权重 DataFrame
# delay_cc: 皮层间延迟 DataFrame
# weight_ext: 外部输入权重 DataFrame
# area_num: 脑区数量
# to_master: 进程间通信队列，发送数据到主进程
# from_master: 进程间通信队列，从主进程接收数据
# done_queue: 进程间通信队列，通知主进程任务完成
# final_queue: 进程间通信队列，发送最终结果到主进程
# ===============================================================
def Part(worker_id, gpu_id,  area_list, all_area, pop_list, NN, SN, weight, delay_cc, area_num, Ind,
         to_master: Queue, from_master: Queue, done_queue: Queue, final_queue: Queue):
    print(f"start proccess {worker_id} on GPU {gpu_id}")
    model = GeNNModel("float", f"GenCODE/worker{worker_id}_on_device{gpu_id}", device_select_method=DeviceSelect.MANUAL, manual_device_id=gpu_id)
    if isinstance(area_list, str):
        area_list = [area_list]
    model.dt = 0.1                                                        # 设置时间步长为 0.1 ms
    model.fuse_postsynaptic_models = True                                 # 开启后突触模型融合以提升性能
    model.default_narrow_sparse_ind_enabled = True                        # 开启窄稀疏连接以节省内存
    model.timing_enabled = True                                           # 开启时间测量以便性能分析
    model.default_var_location = VarLocation.HOST_DEVICE                  # 设置变量默认位置为主机和设备双端
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE  # 设置稀疏连接默认位置为主机和设备双端
    # 设置额外的刺激，为了后续分析模型所设计，目前没用到
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
    Ind_V1 = Ind['V1']
    neuron_group = 0
    # 创建神经元群体
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
                neuron_group += 1
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
                rate = SN[area][pop]['external']['external'] / NN[area][pop] / 3000
                # rate = rate_ext[pop]/1000
                poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                model.add_current_source(area + pop + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)
                neuron_pop.spike_recording_enabled = True

                # print("\tPopulation %s: num neurons:%u, external DC offset:%f" % (popName, pop_size, input[pop]/1000.0))
                total_neurons += pop_size
                neuron_populations[area][pop] = neuron_pop
    total_synapses = 0
    synapse_populations = nested_dict()
    syn_group = 0
    # 创建突触群体
    for areaTar, areaSrc in product(area_list, area_list):
        Ind_ = Ind[areaTar]
        for popTar, popSrc in product(pop_list, pop_list):
            if areaTar == areaSrc:
                factor = Ind_V1[popTar][popSrc] / Ind_[popTar][popSrc] if Ind_[popTar][popSrc] > 0 else 1
            else:
                factor = 1
            wAve = weight[areaTar][popTar][areaSrc][popSrc] / 1000.0 * factor
            wSd = weight[areaTar][popTar][areaSrc][popSrc] / 1000.0 / 10 * factor
            synNum = SN[areaTar][popTar][areaSrc][popSrc]
            tarName = areaTar+popTar
            srcName = areaSrc+popSrc
            synName = srcName+"2"+tarName
            meanDelay=delayMap[areaTar][popTar][areaSrc][popSrc]['ave']
            delay_sd=delayMap[areaTar][popTar][areaSrc][popSrc]['sd']
            max_d=delayMap[areaTar][popTar][areaSrc][popSrc]['max']
            if(synNum>0):
                syn_group += 1
                connect_params = {"num": synNum}
                d_dist = {"mean": meanDelay, "sd": delay_sd, "min": model.dt, "max": max_d}
                total_synapses += synNum
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
    # 代码生成阶段
    time_start = perf_counter()
    model.build()
    time_end = perf_counter()
    build_time = time_end - time_start
    # 分配显存阶段
    time_start = perf_counter()
    model.load(num_recording_timesteps=buffer_size)
    time_end = perf_counter()
    ld_time = time_end - time_start
    print(f"{worker_id} build over")
    # 确保 log 目录存在并清空当前 worker 的 log 文件（truncate）
    os.makedirs("log", exist_ok=True)
    log_path = f"log/worker_{worker_id}.log"
    open(log_path, "w").close()  # 以写模式打开并立即关闭以清空文件内容
    with open(f"log/worker_{worker_id}.log", "a") as f:
        f.write(f"{total_neurons},"
                f"{neuron_group},"
                f"{total_synapses},"
                f"{syn_group},"
                f"{build_time*1000:.2f},"
                f"{ld_time*1000:.2f}\n")
    # 构建 spike_count_buffer 和 inSyn_buffer
    spike_count_buffer, weight_array, prob_array, src_pop_num_array, tar_neu_num_array, R_array  = \
        build_spike_buffer(area_num, NN, SN, delay_cc, weight, dt=model.dt, tar_area_list=area_list,
                           all_area=all_area, pop_list=pop_list, gL=gL)

    cum_src = np.cumsum(src_pop_num_array)  # cumulative counts
    inSyn_buffer = np.zeros(len(R_array), dtype=np.float32)
    cuda.select_device(gpu_id)

    # numba.cuda 相关准备——————————————————————————————————————————————————————————————————————————————————————
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
    # numba.cuda 相关准备——————————————————————————————————————————————————————————————————————————————————————

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

    log_buffer = []
    log_flush_interval = 500  # 每 100 步写一次
    time_start = perf_counter()
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
            # print(f"[Worker {worker_id}] timestep={model.timestep} step_time={(t11-t10)*1000:.3f} ms "
            #       f"update_V={(t9-t6)*1000:.3f} ms update_inSyn={(t4-t3)*1000:.3f} ms "
            #       f"update_decay={(t6-t5)*1000:.3f} ms update_spike={(t2 -t0)*1000:.3f} ms "
            #       f"total_update={(t9-t0)*1000:.3f} ms")
            log_buffer.append(
                f"{model.timestep},"
                f"{(t11-t10)*1000:.3f},"
                f"{(t9-t6)*1000:.3f},"
                f"{(t4-t3)*1000:.3f},"
                f"{(t6-t5)*1000:.3f},"
                f"{(t2-t0)*1000:.3f},"
                f"{(t9-t0)*1000:.3f}"
            )
            if len(log_buffer) >= log_flush_interval:
                with open(f"log/worker_{worker_id}.log", "a") as f:
                    f.write("\n".join(log_buffer) + "\n")
                log_buffer.clear()
        else:
            t10 = perf_counter()
            model.step_time()
            t11 = perf_counter()
            # 子进程本地打印自己想要的时间统计（主进程不再收到这些字段）
            # print(f"[Worker {worker_id}] timestep={model.timestep} step_time={(t11-t10)*1000:.3f} ms ")
            log_buffer.append(
                f"{model.timestep},"
                f"{(t11-t10)*1000:.3f},"
            )
            if len(log_buffer) >= log_flush_interval:
                with open(f"log/worker_{worker_id}.log", "a") as f:
                    f.write("\n".join(log_buffer) + "\n")
                log_buffer.clear()

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
    time_end = perf_counter()
    with open(f"log/worker_{worker_id}.log", "a") as f:
        f.write(f"total_simulation_time,{(time_end - time_start)*1000:.2f} ms\n")
    # 仿真循环结束：一次性把 local_spike_history 发回主进程用于绘图（可能很大）
    final_msg = {"worker_id": worker_id, "final_spike_data": local_spike_history}
    final_queue.put(final_msg)

# ===============================================================
# 合并多个 worker 的 spike data 函数
# 输入参数：
# spike_data_blocks: list of spike_data dicts from multiple workers
# 返回值：
# merged spike_data dict
# ===============================================================
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

# ===============================================================
# 主程序入口
# ===============================================================
if __name__ == "__main__":
    num_gpus = 10      # 使用的 GPU 数量
    # procs_per_gpu = 1  # 每个 GPU 上的进程数
    num_workers = 32   # 总进程数
    split_idx = split_indices(num_workers,num_workers)
    NN, SN, weight, _, delayMap, area_list, pop_list, Ind = prepare()
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
                          NN, SN, weight, delayMap, num_workers, Ind,
                          to_master, from_master, done_queue, final_queue))
        p.start()
        processes.append(p)


    # 主循环
    step = 0
    max_steps = duration_timesteps // buffer_size
    master_log_buffer = []
    log_flush_interval = 100 * num_workers  # 每 100 步写一次
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

            # print(f"[Round {step}] Worker {msg['worker_id']} -> 主进程: "
            #     f"延迟 {latency*1000:.3f} ms, "
            #     f"速度 {speed_MBps:.2f} MB/s, "
            #     f"大小 {data_size/1024:.1f} KB")
            master_log_buffer.append(
                f"{step},"
                f"{msg['worker_id']},"
                f"{latency*1000:.3f},"
                f"{speed_MBps:.2f},"
                f"{data_size/1024:.1f}"
            )
            if len(master_log_buffer) >= log_flush_interval:
                with open(f"log/master.log", "a") as f:
                    f.write("\n".join(master_log_buffer) + "\n")
                master_log_buffer.clear()
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
        # visualize("Test", spike_data_temp, duration=duration, drop=0, neurons_per_group=200, 
        #         group_spacing=20, NeuronNumber=NeuronNumber, vis_content=vis_content)