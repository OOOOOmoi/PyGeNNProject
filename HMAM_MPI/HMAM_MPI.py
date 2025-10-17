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
NUM_THREADS_PER_SPIKE = 1
duration = 1000
DT_MS = 0.1
duration_timesteps = int(round(duration / DT_MS))
ten_percent_timestep = duration_timesteps // 10
buffer_size = 10000

def split_indices(num_areas, num_gpus):
    # 平均分配索引到 num_gpus 个子列表
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_gpus - 1) // num_gpus  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus) if indices[i*chunk_size:(i+1)*chunk_size]]

def Part(worker_id, gpu_id,  area_list, NN, rate_ext, SN, weight, delay_cc, weight_ext,
         to_master: Queue, from_master: Queue):
    print(f"start proccess {worker_id} on GPU {gpu_id}")
    model = GeNNModel("float", f"HMAM_MPI_CODE/worker{worker_id}_on_device{gpu_id}", device_select_method=DeviceSelect.MANUAL, manual_device_id=gpu_id)
    if isinstance(area_list, str):
        area_list = [area_list]
    model.dt = 0.1
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
                        rate = rate_ext.loc[(area, layer, pop)] * 100
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
    flag = 0
    while model.t < duration:
        model.step_time()

        if not model.timestep % buffer_size:
            spike_data_temp = {
                area: {pop: [] for pop in neuron_populations[area].keys()}
                for area in neuron_populations.keys()
            }
            model.pull_recording_buffers_from_device()
            record_spike(neuron_populations, spike_data_temp)

            # 把当前进程的数据发给主进程
            msg = {
                "worker_id": worker_id,
                "spike_data": spike_data_temp,
                "timestamp": time.perf_counter()
            }

            to_master.put(msg)

            # 等待主进程的新指令（同步点）
            msg = from_master.get()
            if msg["type"] == "continue":
                # 可以更新模型参数，例如 rate 或 weight
                updates = msg.get("updates", None)
                if updates:
                    # TODO: 根据 updates 修改模型参数
                    rate = updates["rate"]
                    count = updates["spike_count"]

                    pass
            elif msg["type"] == "stop":
                break

        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print(f"{worker_id}-th Proccess {flag * 10}%")


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

    num_gpus = 1
    procs_per_gpu = 2   # 假设每个GPU跑2个进程
    num_workers = num_gpus * procs_per_gpu
    split_idx = split_indices(4,num_workers)

    to_master_queues = []
    from_master_queues = []
    processes = []

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
                          [area_list[j] for j in split_idx[i]],
                          NN, rate_ext, SN, weight, delay_cc, weight_ext,
                          to_master, from_master))
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
            msg = to_master_queues[i].get()
            recv_time = time.perf_counter()

            latency = recv_time - msg["timestamp"]
            data_size = len(pickle.dumps(msg))
            speed_MBps = data_size / (latency * 1024 * 1024)

            print(f"[Round] Worker {msg['worker_id']} -> 主进程: "
                f"延迟 {latency*1000:.3f} ms, "
                f"速度 {speed_MBps:.2f} MB/s, "
                f"大小 {data_size/1024:.1f} KB")

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
                model_name="HMAM", NeuronNumber=NeuronNumber)