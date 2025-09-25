from mpi4py import MPI
from pygenn import GeNNModel
from pygenn.cuda_backend import DeviceSelect
import numpy as np
import pickle, time
from collections import defaultdict
from scipy.stats import norm
from record import record_spike
from visual import visualize
from config import (
    get_NN, get_SN, get_weight, get_weight_ext, externalRates, get_cc_delay,
    getModelName, remove_dash_from_index_columns, get_ext_rate, net, layer_map
)

duration = 1000
DT_MS = 0.1
duration_timesteps = int(round(duration / DT_MS))
buffer_size = 10000
ten_percent_timestep = duration_timesteps // 10

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_gpus = 8
procs_per_gpu = 2   # 每 GPU 启动的进程数
num_workers = size - 1   # rank 0 是 master

# ---------------- Worker ----------------
def Part(worker_id, gpu_id, area_list, NN, rate_ext, SN, weight, delay_cc, weight_ext, NeuronNumber):
    print(f"[Worker {worker_id}] running on GPU {gpu_id}")

    model = GeNNModel("float", f"HMAM_MPI_CODE/worker{worker_id}_gpu{gpu_id}",
                      device_select_method=DeviceSelect.MANUAL,
                      manual_device_id=gpu_id)
    model.dt = 0.1

    # 这里省略神经网络的 build 部分（和你之前 Part 里相同）
    # ...
    neuron_populations = defaultdict(dict)
    # 构建神经元和突触（与原始 Part 一致）
    # ...

    model.build()
    model.load(num_recording_timesteps=buffer_size)

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

            msg = {
                "worker_id": worker_id,
                "spike_data": spike_data_temp,
                "timestamp": time.perf_counter()
            }
            # send -> master
            comm.gather(msg, root=0)

            # wait broadcast
            ctrl_msg = comm.bcast(None, root=0)
            if ctrl_msg["type"] == "stop":
                break

        if (model.timestep % ten_percent_timestep) == 0:
            flag += 1
            print(f"Worker {worker_id} progress: {flag*10}%")

# ---------------- Master ----------------
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

def Master(NeuronNumber):
    step = 0
    max_steps = duration_timesteps // buffer_size
    all_steps_spike_data = []

    while step < max_steps:
        # gather all worker messages
        gathered = comm.gather(None, root=0)
        # gathered 是一个列表，第 0 个是 master 的 None，其余是 worker 的 msg
        worker_msgs = [m for m in gathered if m is not None]

        # 统计通信延迟和大小
        for msg in worker_msgs:
            recv_time = time.perf_counter()
            latency = recv_time - msg["timestamp"]
            data_size = len(pickle.dumps(msg))
            speed_MBps = data_size / (latency * 1024 * 1024)
            print(f"[Round] Worker {msg['worker_id']} -> Master: "
                  f"延迟 {latency*1000:.3f} ms, 速度 {speed_MBps:.2f} MB/s, "
                  f"大小 {data_size/1024:.1f} KB")

        spike_data_blocks = [m["spike_data"] for m in worker_msgs]
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

        # broadcast 更新
        comm.bcast({"type": "continue", "updates": processed_data}, root=0)
        step += 1

    # stop 所有 worker
    comm.bcast({"type": "stop"}, root=0)
    print("Simulation finished, merging data ...")

    final_spike_data = merge_spike_data(all_steps_spike_data)
    for area, area_dict in final_spike_data.items():
        spike_data_temp = {area: area_dict}
        visualize(suffix="test", spike_data=spike_data_temp, duration=1000,
                  model_name="HMAM", NeuronNumber=NeuronNumber)

# ---------------- Main ----------------
if __name__ == "__main__":
    # 初始化数据（只在 rank=0 做一次，然后广播）
    if rank == 0:
        NN = remove_dash_from_index_columns(get_NN())
        SN, SN_ext = get_SN()
        SN = remove_dash_from_index_columns(SN)
        rate_ext = remove_dash_from_index_columns(get_ext_rate())
        weight, weight_sd = get_weight()
        weight = remove_dash_from_index_columns(weight)
        delay_cc, delay_cc_sd = get_cc_delay()
        delay_cc = remove_dash_from_index_columns(delay_cc)
        weight_ext, weight_ext_sd = get_weight_ext()
        weight_ext = remove_dash_from_index_columns(weight_ext)

        NeuronNumber = defaultdict(dict)
        for area in net["area_list"]:
            for layer in net["layer_list"]:
                for pop in net["population_list"]:
                    if (area, layer, pop) in NN.index:
                        popNum = NN.loc[(area, layer, pop)]
                        NeuronNumber[area][pop+layer_map[layer]] = popNum

        shared_data = (NN, SN, rate_ext, weight, delay_cc, weight_ext, NeuronNumber)
    else:
        shared_data = None

    NN, SN, rate_ext, weight, delay_cc, weight_ext, NeuronNumber = comm.bcast(shared_data, root=0)

    if rank == 0:
        Master(NeuronNumber)
    else:
        worker_id = rank
        gpu_id = (rank-1) % num_gpus
        # 每个 worker 绑定的区域子集 (这里你可以继续做 split_indices 分配 area)
        area_subset = net["area_list"][(worker_id-1) % len(net["area_list"])]
        Part(worker_id, gpu_id, [area_subset], NN, rate_ext, SN, weight, delay_cc, weight_ext, NeuronNumber)
