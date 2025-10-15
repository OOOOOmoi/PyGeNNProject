# hmam_mpi.py
import os
import time
import pickle
from time import perf_counter
from collections import defaultdict
from itertools import product

from mpi4py import MPI
import numpy as np
from scipy.stats import norm
import pandas as pd

# your project imports (assumed available on each node in same path)
from config import (
    expLIF_dict, input, layer_map, vis_content,
    get_NN, get_SN, get_weight, get_weight_ext, externalRates,
    get_cc_delay, getModelName, remove_dash_from_index_columns, get_ext_rate, net
)
from visual import visualize
from record import record_spike

from pygenn import GeNNModel, VarLocation, init_postsynaptic, init_sparse_connectivity, init_weight_update, init_var
from pygenn.cuda_backend import DeviceSelect

# ---------------- simulation settings ----------------
NUM_THREADS_PER_SPIKE = 1
duration = 1000
DT_MS = 0.1
duration_timesteps = int(round(duration / DT_MS))
ten_percent_timestep = duration_timesteps // 10
buffer_size = 1   # 建议不要设太小（1 会严重拖慢并行效率）
# -----------------------------------------------------

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()   # 包含 master
num_workers = max(0, size - 1)

# GPU / per-gpu-process config (可按需修改或从环境读取)
NUM_GPUS = 8           # 每节点的 GPU 数（假设每台机器 GPU 数一致）

# ---------------- helper funcs ----------------
def split_indices(num_areas, num_workkers):
    # 平均分配索引到 num_gpus 个子列表
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_workkers - 1) // num_workkers  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workkers) if indices[i*chunk_size:(i+1)*chunk_size]]

def estimate_offset_master_peer(comm, peer, niter=10):
    offsets = []
    rtts = []
    for i in range(niter):
        t0 = MPI.Wtime()
        comm.send(None, dest=peer, tag=3000+i)   # ping
        t1 = comm.recv(source=peer, tag=3000+i)  # t1 is worker's recv-time (worker sends it)
        t2 = MPI.Wtime()
        rtt = t2 - t0
        offset = (t0 + t2) / 2.0 - t1
        offsets.append(offset)
        rtts.append(rtt)
    # choose offset corresponding to minimal RTT samples to reduce asymmetry effect
    k = max(1, niter//4)
    idx = sorted(range(len(rtts)), key=lambda i: rtts[i])[:k]
    avg_offset = sum(offsets[i] for i in idx) / len(idx)
    return avg_offset

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

# ---------------- Worker Part (几乎照搬你的 Part 逻辑) ----------------
def Part(worker_rank, gpu_id, area_list, NN, rate_ext, SN, weight, delay_cc, weight_ext):
    """
    worker_rank: MPI rank (>=1)
    gpu_id: GPU id on local node to bind (int)
    area_list: list of area names assigned to this worker
    the rest: dataframes passed from master
    """
    print(f"[Worker {worker_rank}] start on GPU {gpu_id}, assigned areas: {area_list}", flush=True)

    model = GeNNModel("float", f"HMAM_MPI_CODE/worker{worker_rank}_gpu{gpu_id}",
                      device_select_method=DeviceSelect.MANUAL,
                      manual_device_id=gpu_id)
    model.dt = DT_MS

    layer_list = net["layer_list"]
    pop_list = net["population_list"]

    lif_init = {"V": init_var("Uniform", {"max": -50.0, "min": -200.0}), "RefracTime": 0.0}
    poisson_init = {"current": 0.0}
    total_neurons = 0
    NeuronNumber_local = defaultdict(dict)
    neuron_populations = defaultdict(dict)

    # create neuron populations assigned to this worker
    for area in area_list:
        for layer in layer_list:
            for pop in pop_list:
                if (area, layer, pop) in NN.index:
                    popName = area + pop + layer_map[layer]
                    popNum = int(NN.loc[(area, layer, pop)])
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
                        rate = rate_ext.loc[(area, layer, pop)] * 1000
                        poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate}
                        model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init)

                        neuron_pop.spike_recording_enabled = True
                        total_neurons += popNum
                        neuron_populations[area][pop + layer_map[layer]] = neuron_pop
    # for update 
    # need sn to all assign areas
    weight_local = weight.loc(axis=1)[area_list,:,:]
    sn_local = SN.loc(axis=1)[area_list,:,:]

    # synapse creation (按原逻辑)
    exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 2})
    inh_curr_init = init_postsynaptic("ExpCurr", {"tau": 5})
    total_synapses = 0
    syn_group_num = 0

    # 注意：原来代码在构建突触时遍历 area_list x area_list -> 这里只考虑 assigned areas.
    for tar_area, src_area in product(area_list, area_list):
        for tar_layer, src_layer in product(layer_list, layer_list):
            for tar_pop, src_pop in product(pop_list, pop_list):
                tar = (tar_area, tar_layer, tar_pop)
                src = (src_area, src_layer, src_pop)
                if tar in SN.index and src in SN.columns:
                    tarName = tar_area + tar_pop + layer_map[tar_layer]
                    srcName = src_area + src_pop + layer_map[src_layer]
                    synName = srcName + "_to_" + tarName
                    synNum = int(SN.loc[tar, src])
                    wAve = weight.loc[tar, src] / 1000.0
                    wSd = wAve / 10.0 / 1000.0
                    if src_area == tar_area:
                        if src_pop == 'E':
                            meanDelay = net['delay_e']; delay_sd = net['delay_e_sd']
                        else:
                            meanDelay = net['delay_i']; delay_sd = net['delay_i_sd']
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
                        static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
                                                                {"g": init_var("NormalClipped", w_dist),
                                                                 "d": init_var("NormalClippedDelay", d_dist)})
                        syn_pop = model.add_synapse_population(synName, matrix_type,
                                                              srcPop, tarPop,
                                                              static_synapse_init, curr_init,
                                                              init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
                        syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / model.dt))
                        if matrix_type == "PROCEDURAL":
                            syn_pop.num_threads_per_spike = NUM_THREADS_PER_SPIKE

    print(f"[Worker {worker_rank}] Building done: neurons {total_neurons}, synapses {total_synapses}, groups {syn_group_num} on GPU {gpu_id}", flush=True)
    model.build()
    model.load(num_recording_timesteps=buffer_size)
    print(f"[Worker {worker_rank}] Model loaded; starting sim loop", flush=True)

    flag = 0
    # simulation loop - note buffer_size timesteps per communication round
    while model.t < duration:
        # step buffer_size times (but model.step_time does dt each call)
        model.step_time()

        if (model.timestep % buffer_size) == 0:
            # pull and record
            model.pull_recording_buffers_from_device()
            spike_data_temp = {area: {pop: [] for pop in neuron_populations[area].keys()} for area in neuron_populations.keys()}
            record_spike(neuron_populations, spike_data_temp)

            # prepare message for master
            msg = {
                "worker_rank": worker_rank,
                "spike_data": spike_data_temp,
                "NeuronNumber": NeuronNumber_local,
                "timestamp_sent": MPI.Wtime()
            }
            # all workers call gather, master will get list
            comm.gather(msg, root=0)

            # wait for master bcast (synchronization)
            ctrl = comm.bcast(None, root=0)
            if ctrl.get("type") == "stop":
                print(f"[Worker {worker_rank}] received STOP", flush=True)
                break
            # else ctrl contains updates that you may apply (TODO)
            updates = ctrl.get("updates", None)
            if updates:
                # apply update logic if needed (example: reading updates["rate"])
                rate_info = updates["rate"]
                pass

        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print(f"[Worker {worker_rank}] progress {flag*10}%", flush=True)

    print(f"[Worker {worker_rank}] simulation finished.", flush=True)


# ---------------- Master ----------------
def Master(NN, SN, rate_ext, weight, delay_cc, weight_ext, NeuronNumber_global):
    max_steps = duration_timesteps // buffer_size
    step = 0
    all_steps_spike_data = []
    offsets = {peer: estimate_offset_master_peer(comm, peer, niter=12) for peer in range(1, size)}
    print("[MASTER] enter main loop", flush=True)
    while step < max_steps:
        step += 1
        # collect worker messages via gather: each worker did comm.gather(msg, root=0)
        gathered = comm.gather(None, root=0)  # root collects list: [None, msg_from_rank1, msg_from_rank2, ...]
        # filter out None entries
        worker_msgs = [m for m in gathered if m is not None]

        # compute latency / size and collect spike blocks
        spike_blocks = []
        for msg in worker_msgs:
            recv_time = MPI.Wtime()
            ts_worker = msg["timestamp_sent"]
            corrected = ts_worker + offsets[msg["worker_rank"]]
            latency = recv_time - corrected
            data_size = len(pickle.dumps(msg.get("spike_data", {})))
            speed = data_size / (latency * 1024 * 1024) if (latency and latency > 0) else float('inf')
            print(f"[MASTER][step {step}] from worker {msg.get('worker_rank')} - latency {latency*1000 if latency else None:.3f} ms, size {data_size/1024:.1f} KB, speed {speed:.2f} MB/s", flush=True)
            spike_blocks.append(msg.get("spike_data"))

        # merge and compute processed_data
        all_spike_data = merge_spike_data(spike_blocks)
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
                num_neurons = int(NeuronNumber_global[area][pop])
                spike_rate = spike_count / num_neurons / duration * 1000.0
                processed_data["rate"][area][pop] = spike_rate
                processed_data["spike_count"][area][pop] = spike_count
                # print(f"[MASTER] {area} {pop} -> rate {spike_rate:.3f} Hz, count {spike_count}", flush=True)

        # broadcast updates to all workers (bcast blocks until all workers call bcast)
        ctrl_msg = {"type": "continue", "updates": processed_data, "step": step}
        comm.bcast(ctrl_msg, root=0)

    # end simulation -> tell workers to stop
    comm.bcast({"type": "stop"}, root=0)
    print("[MASTER] finished main loop, merging all steps and visualizing", flush=True)

    # merge full history and visualize (master only)
    final_spike_data = merge_spike_data(all_steps_spike_data)
    # optionally visualize per area to avoid huge memory spike
    for area, area_dict in final_spike_data.items():
        spike_data_chunk = {area: area_dict}
        visualize(suffix="mpi_final", spike_data=spike_data_chunk, duration=duration, model_name="HMAM", NeuronNumber=NeuronNumber_global)

    print("[MASTER] done.", flush=True)


# ---------------- main ----------------
if __name__ == "__main__":
    # load static config on master and broadcast to all
    if rank == 0:
        NN = remove_dash_from_index_columns(get_NN())
        SN, SN_ext = get_SN()
        SN = remove_dash_from_index_columns(SN)
        SN_ext = remove_dash_from_index_columns(SN_ext)
        rate_ext = remove_dash_from_index_columns(get_ext_rate())
        weight, weight_sd = get_weight()
        weight = remove_dash_from_index_columns(weight)
        weight_sd = remove_dash_from_index_columns(weight_sd)
        delay_cc, delay_cc_sd = get_cc_delay()
        delay_cc = remove_dash_from_index_columns(delay_cc)
        delay_cc_sd = remove_dash_from_index_columns(delay_cc_sd)
        weight_ext, weight_ext_sd = get_weight_ext()
        weight_ext = remove_dash_from_index_columns(weight_ext)
        weight_ext_sd = remove_dash_from_index_columns(weight_ext_sd)

        # build global NeuronNumber for visualize
        NeuronNumber_global = defaultdict(dict)
        area_list = net["area_list"]
        area_list = [s.replace("-", "") for s in area_list]
        layer_list = net["layer_list"]
        pop_list = net["population_list"]
        for area in area_list:
            for layer in layer_list:
                for pop in pop_list:
                    if (area, layer, pop) in NN.index:
                        popNum = int(NN.loc[(area, layer, pop)])
                        NeuronNumber_global[area][pop + layer_map[layer]] = popNum

        shared = (NN, SN, rate_ext, weight, delay_cc, weight_ext, NeuronNumber_global, area_list)
    else:
        shared = None

    # broadcast shared data (master -> all)
    NN, SN, rate_ext, weight, delay_cc, weight_ext, NeuronNumber_global, area_list = comm.bcast(shared, root=0)

    # compute area splits among workers (global)
    split_idx = split_indices(68, num_workers)  # splits[i] assigned to worker rank=i+1

    if rank == 0:
        # master main
        Master(NN, SN, rate_ext, weight, delay_cc, weight_ext, NeuronNumber_global)
    else:
        # worker
        worker_rank = rank
        # map to gpu id (assume rank distribution per node is contiguous)
        local_gpu_id = (rank - 1) % NUM_GPUS
        assigned_areas = [area_list[j-1] for j in split_idx[worker_rank - 1]]
        # call Part to build model & run
        Part(worker_rank, local_gpu_id, assigned_areas, NN, rate_ext, SN, weight, delay_cc, weight_ext)
