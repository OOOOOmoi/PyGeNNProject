from config import get_cc_delay, get_NN, get_SN, get_weight, remove_dash_from_index_columns,\
    get_weight_ext, get_ext_rate, net, layer_map
import numpy as np
from time import perf_counter
import pandas as pd
import scipy.io as sio
area_list = net["area_list"]
area_list = [s.replace("-", "") for s in area_list]
layer_list = net["layer_list"]
pop_list = net["population_list"]
if isinstance(area_list, str):
    area_list = [area_list]

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
                for src_area in all_area[1:area_num+1]:
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


def split_indices(num_areas, num_workkers):
    # 平均分配索引到 num_gpus 个子列表
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_workkers - 1) // num_workkers  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workkers) if indices[i*chunk_size:(i+1)*chunk_size]]
NN = get_NN()
# 计算每个 area 中 I 型神经元的比例

area_vals = NN.index.get_level_values(0).to_numpy()
pop_vals = NN.index.get_level_values(2).to_numpy()
vals = NN.to_numpy() if hasattr(NN, "to_numpy") else NN.values

areas = np.unique(area_vals)
i_prop = {}
for a in areas:
    mask_area = area_vals == a
    total = vals[mask_area].sum()
    if total == 0:
        i_prop[a] = 0.0
        continue
    mask_i = mask_area & (pop_vals == "I")
    i_count = vals[mask_i].sum()
    i_prop[a] = float(i_count) / float(total)

I_prop_series = pd.Series(i_prop)
# 将 I_prop_series 保存为 pkl
out_path = "/home/yangjinhao/PyGenn/HMAM_MPI/I_prop_series.pkl"
I_prop_series.to_pickle(out_path)
print("Saved I_prop_series to", out_path)
NN = remove_dash_from_index_columns(NN)
NN_area = NN.groupby(level=0).sum()
NN_area_dict = NN_area.to_dict()
NN_area_ordered = {area: NN_area_dict.get(area, 0) for area in area_list}
sio.savemat("NN_area.mat", {"NN_area": NN_area_ordered})
SN, SN_ext = get_SN()
SN = remove_dash_from_index_columns(SN)
weight, weight_sd = get_weight()
weight = remove_dash_from_index_columns(weight)
delay_cc, delay_cc_sd = get_cc_delay()
delay_cc = remove_dash_from_index_columns(delay_cc)


t_start = perf_counter()
area_num = 2
buffer, weight_array, prob_array, src_pop_num_array, tar_neu_num_array, R_array =\
    build_spike_buffer(area_num, NN, SN, delay_cc, weight, dt=0.1, tar_area_list=[area_list[1]], net=net)
t_end = perf_counter()
print(f"Build spike buffer time: {t_end - t_start:.4f} seconds")

for k in buffer.keys():
    print("Buffer key:", k, "Buffer size:", len(buffer[k]))

current_step = 0
# 1. 将新 spikes 加入 delay buffer
t_start = perf_counter()
for (tar, src), buf in buffer.items():
    spike_count = 10  # 当前时间步源群体的 spike 数
    buf[current_step % len(buf)] += spike_count
t_end = perf_counter()
print(f"Add spikes to buffer time: {t_end - t_start:.4f} seconds")

# 2. 取出到达的 spikes 并清零
t_start = perf_counter()
spike_array = []
for (src, tar), buf in buffer.items():
    deliver_step = (current_step - len(buf)) % len(buf)
    arriving_spikes = buf[deliver_step]
    spike_array.append(arriving_spikes)
    # update_membrane_potential(tar, arriving_spikes)
    buf[deliver_step] = 0.0
t_end = perf_counter()
print(f"Retrieve spikes from buffer time: {t_end - t_start:.4f} seconds")
inSyn_buffer = np.zeros(NN_area[area_list[0]], dtype=np.float32)
cum_src = np.cumsum(src_pop_num_array)  # cumulative counts
for sc_idx, sc in enumerate(spike_array):
    if sc == 0:
        continue
    group_idx = int(np.searchsorted(cum_src, sc_idx + 1))
    start_id = int(np.sum(tar_neu_num_array[:group_idx])) if group_idx > 0 else 0
    group_neu_count = int(tar_neu_num_array[group_idx]) if group_idx < len(tar_neu_num_array) else 0
    prob = prob_array[sc_idx]
    while sc >= 0:
        # 按二项分布从 group_neu_count 个目标神经元中抽取命中数 k，再随机选择 k 个目标神经元并累加权重
        k = np.random.binomial(group_neu_count, prob)
        if k > 0:
            if k >= group_neu_count:
                post_idxs = np.arange(start_id, start_id + group_neu_count, dtype=int)
            else:
                choices = np.random.choice(group_neu_count, size=k, replace=False)
                post_idxs = start_id + choices.astype(int)
            inSyn_buffer[post_idxs] += weight_array[sc_idx]
        sc -= 1
# 找到非零元素的索引和值
nonzero_idx = np.flatnonzero(inSyn_buffer)          # 1D 索引数组
nonzero_vals = inSyn_buffer[nonzero_idx]           # 对应的值

# 示例：打印或进一步处理
if nonzero_idx.size:
    for idx, val in zip(nonzero_idx, nonzero_vals):
        print("neuron", idx, "value", val)
else:
    print("inSyn_buffer 中没有非零元素")
weight_ext, weight_ext_sd = get_weight_ext()
rate_ext = get_ext_rate()
rate_ext = remove_dash_from_index_columns(rate_ext)

split_idx = split_indices(68, 34)  # splits[i] assigned to worker rank=i+1
for i in range(34):
    if i==0:
        continue   # rank 0 不处理
    assigned_areas = [area_list[j-1] for j in split_idx[i]]
    weight_local = weight.loc(axis=1)[assigned_areas,:,:]