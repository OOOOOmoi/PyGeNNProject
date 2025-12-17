# ////////////////////////////////////////////////////////////////////
# //                          _ooOoo_                               //
# //                         o8888888o                              //
# //                         88" . "88                              //
# //                         (| ^_^ |)                              //
# //                         O\  =  /O                              //
# //                      ____/`---'\____                           //
# //                    .'  \\|     |//  `.                         //
# //                   /  \\|||  :  |||//  \                        //
# //                  /  _||||| -:- |||||-  \                       //
# //                  |   | \\\  -  /// |   |                       //
# //                  | \_|  ''\---/''  |   |                       //
# //                  \  .-\__  `-`  ___/-. /                       //
# //                ___`. .'  /--.--\  `. . ___                     //
# //              ."" '<  `.___\_<|>_/___.'  >'"".                  //
# //            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
# //            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
# //      ========`-.____`-.___\_____/___.-`____.-'========         //
# //                           `=---='                              //
# //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
# //         佛祖保佑       永无BUG     永不修改                       //
# ////////////////////////////////////////////////////////////////////

import os
import sys
import time
import pickle
from config import expLIF_dict, input, layer_map, vis_content, \
    get_NN, get_SN, get_weight, get_weight_ext, externalRates, get_cc_delay, \
    getModelName, remove_dash_from_index_columns, get_ext_rate, net
from visual import visualize
from record import record_spike, save_spike
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
MAX_SHARED_BINS = 1024
duration = 500
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
        if group_idx >= len(tar_neu_num_array):
            continue

        group_neu_count = tar_neu_num_array[group_idx]
        if group_neu_count <= 0:
            continue

        start_id = np.sum(tar_neu_num_array[:group_idx]) if group_idx > 0 else 0

        prob = prob_array[sc_idx]
        weight = weight_array[sc_idx]

        # 合并sc次binomial
        n_trials = sc * group_neu_count
        k_total = np.random.binomial(n_trials, prob)

        if k_total <= 0:
            continue

        # 不允许超过 group_size
        if k_total >= group_neu_count:
            # 将 k_total 均匀分配到 group_neu_count 上：
            # 每个 neuron + q 次，剩余 r 个随机分配 +1 次
            q = k_total // group_neu_count
            r = k_total - q * group_neu_count  # same as k_total % group_neu_count

            # 首先给每个 neuron + q * weight
            if q > 0:
                add = q * weight
                for post in range(group_neu_count):
                    inSyn_buffer[start_id + post] += add

            # 然后随机选择 r 个不同 neuron，各 +1 * weight
            if r > 0:
                # 生成索引数组并做部分 Fisher-Yates 前 r 步
                idxs = np.arange(group_neu_count)
                for j in range(r):
                    # 选择 [j, group_neu_count)
                    rpos = j + np.random.randint(0, group_neu_count - j)
                    tmp = idxs[j]
                    idxs[j] = idxs[rpos]
                    idxs[rpos] = tmp
                    inSyn_buffer[start_id + idxs[j]] += weight

        else:
            # k_total < group_neu_count：从 group_neu_count 中选 k_total 个不同 neuron，每个 +1
            idxs = np.arange(group_neu_count)
            for j in range(k_total):
                rpos = j + np.random.randint(0, group_neu_count - j)
                tmp = idxs[j]
                idxs[j] = idxs[rpos]
                idxs[rpos] = tmp
                inSyn_buffer[start_id + idxs[j]] += weight

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

def fast_update_inSyn_cupy(
    spike_array_gpu,      # cupy.int32[:], length = n_src
    cum_src_gpu,          # cupy.int32[:], length = n_groups  (cumsum of src_pop_num_array)
    tar_neu_num_gpu,      # cupy.int32[:], length = n_groups
    prob_gpu,             # cupy.float32[:], length = n_src
    weight_gpu,           # cupy.float32[:], length = n_src
    inSyn_gpu,            # cupy.float32[:], length = total_target_neurons (will be updated in-place)
    rng_state=None,       # not used, but kept for API compatibility
    poisson_threshold=2048
):
    """
    GPU-side implementation using cupy vectorized RNG.
    This will modify `inSyn_gpu` in-place.
    NOTE: This function loops over source indices on the host (Python) level,
    but all heavy work (random draws / permutation / index updates) stays on device.
    This tradeoff is acceptable when n_src is moderate (e.g. thousands).
    """
    # move small control arrays to host only if needed (but keep device arrays for ops)
    n_src = int(spike_array_gpu.size)
    # we will read some arrays to host for indexing decisions (cheap)
    spike_host = spike_array_gpu.get()  # small-ish: length n_src
    cum_src_host = cum_src_gpu.get()
    tar_neu_host = tar_neu_num_gpu.get()
    # we will use cupy ops for heavy random draws and permute
    for sc_idx in range(n_src):
        sc = int(spike_host[sc_idx])
        if sc <= 0:
            continue

        # find group index: searchsorted on cum_src_host
        # target is sc_idx+1 as in your numba code
        target = sc_idx + 1
        # binary search (numpy-style)
        group_idx = int(cp.searchsorted(cum_src_gpu, target).get())  # using cupy searchsorted then get index -> small cost

        if group_idx >= len(tar_neu_host):
            continue
        group_neu_count = int(tar_neu_host[group_idx])
        if group_neu_count <= 0:
            continue

        # start id (sum of previous tar_neu)
        start_id = int(tar_neu_host[:group_idx].sum()) if group_idx > 0 else 0

        prob = float(prob_gpu[sc_idx].get())
        weight = float(weight_gpu[sc_idx].get())

        # total trials (sc repeats)
        n_trials = sc * group_neu_count

        # sample k_total: if n_trials small => binomial; else use poisson approx
        if n_trials < poisson_threshold:
            # cupy supports binomial with vector args; do a single scalar draw via cupy.random.binomial
            k_total = int(cp.random.binomial(n_trials, prob))
        else:
            lam = float(n_trials) * prob
            # poisson sometimes unstable for very large lambda; clip if necessary
            k_total = int(cp.random.poisson(lam))

        if k_total <= 0:
            continue

        # clip k_total to at most group_neu_count * something? your original caps at group_neu_count only in distribution logic
        if k_total >= group_neu_count:
            q = k_total // group_neu_count
            r = k_total - q * group_neu_count
            if q > 0:
                # add q * weight to all neurons in group -> vector add slice
                # inSyn_gpu[start_id : start_id+group_neu_count] += q * weight
                cp.add(inSyn_gpu[start_id:start_id+group_neu_count],
                       cp.float32(q * weight),
                       out=inSyn_gpu[start_id:start_id+group_neu_count])

            if r > 0:
                # choose r unique indices uniformly from [0, group_neu_count)
                # use permutation and take first r
                perm = cp.random.permutation(group_neu_count)[:r]
                # atomic add: cupy advanced indexing will produce a gather+scatter; to avoid duplicate index issues,
                # we use bincount with weights for accumulating per-position increments then add slice once.
                # build per-group increments:
                if r == 1:
                    # simpler fast path
                    idx = int(perm[0])
                    cp.add(inSyn_gpu[start_id + idx], cp.float32(weight), out=inSyn_gpu[start_id + idx])
                else:
                    # create counts per position (mostly 0/1)
                    counts = cp.bincount(perm, minlength=group_neu_count).astype(cp.float32)
                    if counts.size > 0:
                        # multiply by weight and add to slice
                        cp.add(inSyn_gpu[start_id:start_id+group_neu_count],
                               counts * cp.float32(weight),
                               out=inSyn_gpu[start_id:start_id+group_neu_count])
        else:
            # k_total < group_neu_count: choose k_total unique posts
            if k_total == 1:
                # draw one index
                post = int(cp.random.randint(0, group_neu_count))
                cp.add(inSyn_gpu[start_id + post], cp.float32(weight), out=inSyn_gpu[start_id + post])
            else:
                perm = cp.random.permutation(group_neu_count)[:k_total]
                counts = cp.bincount(perm, minlength=group_neu_count).astype(cp.float32)
                cp.add(inSyn_gpu[start_id:start_id+group_neu_count],
                       counts * cp.float32(weight),
                       out=inSyn_gpu[start_id:start_id+group_neu_count])

    # nothing to return; inSyn_gpu updated in-place
    return None

def decay_cupy(inSyn_gpu, R_gpu, dt):
    """
    Implement decay + IR computation with cupy vector ops.
    original:
        inSyn[i] *= exp(-dt/4)
        IR[i] = inSyn[i] * R[i]
        inSyn[i] *= exp(-dt/2)
    We implement exactly the same ordering.
    """
    # equivalent vector ops on GPU
    a = cp.exp(-dt / 4.0, dtype=cp.float32)
    b = cp.exp(-dt / 2.0, dtype=cp.float32)
    # do operations
    inSyn_gpu *= a
    IR_gpu = inSyn_gpu * R_gpu   # new device array
    inSyn_gpu *= b
    return IR_gpu  # return IR on device


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

import cupy as cp
import numpy as np

# ===============================================================
#  CUDA C RawKernel : fast_update_inSyn_gpu
# ===============================================================
kernel_code = r'''
extern "C" {

#define MAX_THREADS 1024
#define MAX_SHARED_BINS 1024

// rotate left for 64-bit
__device__ inline unsigned long long rotl(const unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoroshiro128+ step: state is two uint64s stored at states[2*idx], states[2*idx+1]
// returns new random uint64 and updates state in-place
__device__ inline unsigned long long xoroshiro128p_next(unsigned long long *states, int idx) {
    unsigned long long s0 = states[2*idx];
    unsigned long long s1 = states[2*idx + 1];
    unsigned long long result = s0 + s1;

    s1 ^= s0;
    states[2*idx] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
    states[2*idx + 1] = rotl(s1, 36);
    return result;
}

// uniform float32 in [0,1)
__device__ inline float xoroshiro128p_uniform_float32(unsigned long long *states, int idx) {
    unsigned long long r = xoroshiro128p_next(states, idx);
    const unsigned long long UINT23_MASK = (1ULL << 23) - 1ULL;
    unsigned int x = (unsigned int)((r >> 41) & UINT23_MASK);
    return (float)x / (float)(1u << 23);
}


// The kernel
__global__ void fast_update_inSyn_gpu(
    const int *spike_array,         // int32[n_src]
    const int *cum_src,             // int32[n_group]
    const int *tar_neu_num_array,   // int32[n_group]
    const float *prob_array,        // float32[n_src]
    const float *weight_array,      // float32[n_src]
    float *inSyn_buffer,            // float32[sum_tar_neu]
    unsigned long long *rng_states, // uint64[2 * n_rng_states]
    const int n_groups              // cum_src length
) {
    const int sc_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    if (sc_idx >= gridDim.x) return;

    int sc = spike_array[sc_idx];
    if (sc <= 0) return;

    // ===== Binary search: find group_idx such that cum_src[group_idx] >= sc_idx+1 =====
    int target = sc_idx + 1;
    int left = 0, right = n_groups - 1;
    int group_idx = 0;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (target <= cum_src[mid]) {
            group_idx = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // ===== compute start_id =====
    int start_id = 0;
    for (int i = 0; i < group_idx; ++i)
        start_id += tar_neu_num_array[i];

    int group_neu_count = tar_neu_num_array[group_idx];
    float prob = prob_array[sc_idx];
    float weight = weight_array[sc_idx];

    if (group_neu_count <= 0) return;

    int n_trials = sc * group_neu_count;

    // shared
    __shared__ int shared_hits[MAX_THREADS];
    __shared__ int total_hits_shared;

    int total_hits = 0;

    // ===== Bernoulli (exact) when trials < 2048 =====
    if (n_trials < 2048) {
        int trials_per_thread = (n_trials + n_threads - 1) / n_threads;
        int base = tid * trials_per_thread;

        int local_hits = 0;
        int rng_base = sc_idx * n_threads;

        for (int i = 0; i < trials_per_thread; ++i) {
            int t = base + i;
            if (t < n_trials) {
                int ridx = rng_base + (t % n_threads);
                float u = xoroshiro128p_uniform_float32(rng_states, ridx);
                if (u < prob) local_hits += 1;
            }
        }

        shared_hits[tid] = local_hits;
        __syncthreads();

        int stride = n_threads >> 1;
        while (stride > 0) {
            if (tid < stride)
                shared_hits[tid] += shared_hits[tid + stride];
            __syncthreads();
            stride >>= 1;
        }

        if (tid == 0) total_hits_shared = shared_hits[0];
        __syncthreads();
        total_hits = total_hits_shared;
    }
    else {
        // ===== Poisson approx =====
        if (tid == 0) {
            float lam = (float)sc * (float)group_neu_count * prob;
            float L = expf(-lam);
            float p_acc = 1.0f;
            int k = 0;

            int ridx = sc_idx * n_threads;  // 采用 tid=0 的 rng 基
            while (p_acc > L) {
                ++k;
                float u = xoroshiro128p_uniform_float32(rng_states, ridx);
                p_acc *= u;
            }
            total_hits_shared = k - 1;
        }
        __syncthreads();
        total_hits = total_hits_shared;
    }

    if (total_hits <= 0) return;

    // ===== small group: use shared histogram =====
    if (group_neu_count <= MAX_SHARED_BINS) {
        __shared__ float shared_bins[MAX_SHARED_BINS];

        for (int i = tid; i < group_neu_count; i += n_threads)
            shared_bins[i] = 0.0f;
        __syncthreads();

        int hits_per_thread = (total_hits + n_threads - 1) / n_threads;
        int base = tid * hits_per_thread;
        int rng_base = sc_idx * n_threads;

        for (int i = 0; i < hits_per_thread; ++i) {
            int h = base + i;
            if (h < total_hits) {
                int ridx = rng_base + (i % n_threads);
                float u = xoroshiro128p_uniform_float32(rng_states, ridx);
                int post = (int)(u * (float)group_neu_count);
                if (post >= group_neu_count) post = group_neu_count - 1;
                atomicAdd(&shared_bins[post], weight);
            }
        }

        __syncthreads();

        for (int i = tid; i < group_neu_count; i += n_threads) {
            float val = shared_bins[i];
            if (val != 0.f)
                atomicAdd(&inSyn_buffer[start_id + i], val);
        }
    }
    else {
        // ===== large group: atomic to global directly =====
        int hits_per_thread = (total_hits + n_threads - 1) / n_threads;
        int base = tid * hits_per_thread;
        int rng_base = sc_idx * n_threads;

        for (int i = 0; i < hits_per_thread; ++i) {
            int h = base + i;
            if (h < total_hits) {
                int ridx = rng_base + (i % n_threads);
                float u = xoroshiro128p_uniform_float32(rng_states, ridx);
                int post = (int)(u * (float)group_neu_count);
                if (post >= group_neu_count) post = group_neu_count - 1;
                atomicAdd(&inSyn_buffer[start_id + post], weight);
            }
        }
    }
}

} // extern "C"
'''


# ===============================================================
#  RNG 初始化 (splitmix64)
# ===============================================================
def init_xoroshiro_states(n_states, seed=1234567):
    """生成 xoroshiro128+ 的 state 数组，形状为 (2*n_states,) uint64"""
    def splitmix64(x):
        x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31), x

    h = np.empty((2 * n_states,), dtype=np.uint64)
    x = seed
    for i in range(n_states):
        v1, x = splitmix64(x)
        v2, x = splitmix64(x)
        h[2*i] = v1
        h[2*i+1] = v2
    return cp.asarray(h)


def Part(worker_id, gpu_id,  area_list, NN, rate_ext, SN, weight, delay_cc, weight_ext, area_num,
         to_master: Queue, from_master: Queue, done_queue: Queue, final_queue: Queue):
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
                        # print("creating neuron group {popName} with {popNum} neurons".format(popName=popName, popNum=popNum))
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
                        rate = rate_ext.loc[(area, layer, pop)] * 5
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

    # numba.cuda 相关准备——————————————————————————————————————————————————————————————————————————————————————
    cuda.select_device(gpu_id)
    d_cum_src   = cuda.to_device(cum_src.astype(np.int32))
    d_tar_num   = cuda.to_device(tar_neu_num_array.astype(np.int32))
    d_prob      = cuda.to_device(prob_array.astype(np.float32))
    d_weight    = cuda.to_device(weight_array.astype(np.float32))
    d_inSyn     = cuda.to_device(inSyn_buffer.astype(np.float32))
    d_R         = cuda.to_device(R_array.astype(np.float32))
    d_IR = cuda.device_array_like(d_inSyn)  # 用作中间 IR 存储

    # 更新inSyn的 kernel 配置
    d_spike_array = cuda.device_array(len(spike_count_buffer), dtype=np.int32)
    threads = 128  # 或 256，视 GPU 而定；128 是比较保守的起点
    blocks = len(spike_count_buffer)  # 一个 block per sc
    rng_states = create_xoroshiro128p_states(blocks * threads, seed=1234)

    # decay的kernel配置
    N_inSyn = len(inSyn_buffer)  # 先前已知
    IR = np.empty_like(inSyn_buffer, dtype=np.float32)
    threads_per_block = 128
    blocks_per_grid = (N_inSyn + threads_per_block - 1) // threads_per_block
    # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    # cupy 相关准备———————————————————————————————————————————————————————————————————————————————————————
    # cp.cuda.Device(gpu_id).use()
    # d_cum_src = cp.array(cum_src.astype(np.int32))
    # d_tar_num = cp.array(tar_neu_num_array.astype(np.int32))
    # d_prob = cp.array(prob_array.astype(np.float32))
    # d_weight = cp.array(weight_array.astype(np.float32))
    # d_inSyn = cp.array(inSyn_buffer.astype(np.float32))   # will be updated in-place
    # d_R = cp.array(R_array.astype(np.float32))
    
    # # 初始化 RNG states for cupy kernel
    # block_size = 256
    # n_blocks = len(inSyn_buffer)
    # n_states = n_blocks * block_size
    # rng_states = init_xoroshiro_states(n_states, seed=202503)
    # fast_update = cp.RawKernel(kernel_code, 'fast_update_inSyn_gpu')

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
    total_neurons_all = offset

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

            # 2) 更新 inSyn (cupy 加速版)
            # t3 = perf_counter()
            # d_spike_array = cp.array(spike_array.astype(np.int32))
            # fast_update_inSyn_cupy(
            #     d_spike_array,
            #     d_cum_src,
            #     d_tar_num,
            #     d_prob,
            #     d_weight,
            #     d_inSyn
            # )
            # cp.cuda.runtime.deviceSynchronize()
            # t4 = perf_counter()

            # 2) 更新 inSyn (cupy RawKernel 版)
            # t3 = perf_counter()
            # d_spike_array = cp.array(spike_array.astype(np.int32))
            # args = (
            #     d_spike_array,
            #     d_cum_src,
            #     d_tar_num,
            #     d_prob,
            #     d_weight,
            #     d_inSyn,
            #     rng_states,
            #     np.int32(len(src_pop_num_array))
            # )
            # fast_update(
            #     (n_blocks,), (block_size,),
            #     args
            # )
            # t4 = perf_counter()

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
            d_inSyn.copy_to_host(inSyn_buffer)
            # inSyn_buffer *= np.exp(-model.dt / 4)
            # IR = inSyn_buffer * R_array
            # IR = np.asarray(IR)
            # inSyn_buffer *= np.exp(-model.dt / 2)

            # cupy 版 decay + IR
            # d_IR = decay_cupy(d_inSyn, d_R, model.dt)
            # cp.cuda.runtime.deviceSynchronize()
            # IR = cp.asnumpy(d_IR)
            t6 = perf_counter()

            # 5) 将 IR 应用到各 population（注意 slice 的 offset 必须与 build_spike_buffer 中的 group 划分一致）
            # for (area, pop), (start, size) in pop_offsets.items():
            #     if size == 0:
            #         continue
            #     neu_pop = neuron_populations[area][pop]
            #     neu_pop.vars["V"].pull_from_device()
            #     neu_pop.vars["RefracTime"].pull_from_device()
            #     array_V_tmp = neu_pop.vars["V"].current_view.copy()
            #     array_tref_tmp = neu_pop.vars["RefracTime"].current_view.copy()
            #     ir_segment = IR[start : start + size]   # 这是一个 view（不复制，Numpy 切片是 view）
            #     array_segment = array_V[start : start + size]
            #     dv = (ir_segment - array_segment + neu_pop.params["Vrest"].value) * model.dt / neu_pop.params["TauM"].value
            #     mask = (array_tref_tmp <= 0)
            #     dv[~mask] = 0.0
            #     array_V_tmp += dv
            #     neu_pop.vars["V"].current_view[:] = array_V_tmp
            #     neu_pop.vars["V"].push_to_device()
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
    NN_area = NN.groupby(level=0).sum()
    NN_area_dict = NN_area.to_dict()
    NN_area_ordered = {area: NN_area_dict.get(area, 0) for area in area_list}
    NeuronNumber = defaultdict(dict)
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in NN.index:
                    popNum = NN.loc[(area, layer, pop)]
                    NeuronNumber[area][pop+layer_map[layer]] = popNum

    # for area in area_list:
    #     factor = 1.00001e09/68/NN_area_ordered[area]
    #     for layer in layer_list:
    #         for pop in pop_list:
    #             if (area, layer, pop) in NN.index:
    #                 NN.loc[(area, layer, pop)] = int(NN.loc[(area, layer, pop)] * factor)
    num_gpus = 10
    procs_per_gpu = 1
    num_workers = 8
    split_idx = split_indices(num_workers,num_workers)
    # split_idx = [[2], [3], [5], [13]]
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
        assigned_areas = [area_list[j-1+60] for j in split_idx[i]]
        p = Process(target=Part,
                    args=(i,
                          9-gpu_id,
                          assigned_areas,
                          NN, rate_ext, SN, weight, delay_cc, weight_ext, num_workers,
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
        visualize(suffix="test", spike_data=spike_data_temp, duration=1000,
                model_name="HMAM", NeuronNumber=NeuronNumber, drop=0)