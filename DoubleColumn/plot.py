import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# ------------------- 配置 -------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_folder=os.path.join(current_dir, "output")
output_folder=os.path.join(current_dir, "figures")
os.makedirs(output_folder, exist_ok=True)

bin_size = 10  # ms
smooth_window = 5  # 平滑窗口

# 只保留绘制从100ms开始之后的数据
cut_start_time = 100.0  

# ------------------- 平滑函数 -------------------
def moving_average(x, window_size=5):
    """简单滑动平均"""
    return np.convolve(x, np.ones(window_size) / window_size, mode='same')

# ------------------- 读取所有数据 -------------------
# 组织成：spike_data[area][pop] = ndarray(N,2)
spike_data = defaultdict(dict)

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

for file in csv_files:
    filename = os.path.basename(file)
    name, _ = os.path.splitext(filename)
    area, pop, *_ = name.split("_")

    data = np.loadtxt(file, delimiter=",", skiprows=1)  # 第一行是header
    if data.ndim == 1:
        data = data.reshape(1, -1)  # 兼容只有一个spike的情况
    spike_data[area][pop] = data

# ------------------- 绘图 -------------------
for area, pop_dict in spike_data.items():
    print(f"Plotting for {area}...")
    
    ordered_pops = sorted(pop_dict.keys())  # pop名字按顺序排一下
    
    # ------- Raster Plot -------
    fig_raster, ax_raster = plt.subplots(figsize=(12, 6))

    start_id = 0
    pop_sizes = {}
    flag=True
    for pop in ordered_pops:
        data = pop_dict[pop]
        spike_times = data[:, 0]
        spike_ids = data[:, 1]

        # 去掉cut_start_time之前的
        mask = spike_times >= cut_start_time
        spike_times = spike_times[mask]
        spike_ids = spike_ids[mask]

        # ⭐ 只保留ID在前5%的spike
        if spike_ids.size > 0:
            unique_ids = np.unique(spike_ids)
            n_top = max(1, int(len(unique_ids) * 0.05))  # 至少保留1个神经元
            top_ids = np.sort(unique_ids)[:n_top]  # 取ID最小的前5%
            mask_id = np.isin(spike_ids, top_ids)
            spike_times = spike_times[mask_id]
            spike_ids = spike_ids[mask_id]

        # 画图
        color = 'red' if flag else 'blue'
        ax_raster.scatter(spike_times, spike_ids + start_id, s=5, edgecolors="none", color=color)
        flag= not flag
        # 更新ID偏移
        max_id = int(spike_ids.max()) if spike_ids.size > 0 else 0
        pop_sizes[pop] = max_id + 100
        start_id += pop_sizes[pop]

    ax_raster.set_title(f"Raster Plot for {area}")
    ax_raster.set_xlabel("Time [ms]")
    ax_raster.set_ylabel("Neuron ID")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{area}_raster_plot.jpg")
    plt.close(fig_raster)

    # ------- Firing Rate Plot -------
    fig_rate, (ax_rate, ax_bar) = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [2, 1]})

    # 重新计算时间范围
    all_times = np.concatenate([d[:,0] for d in pop_dict.values()])
    duration_ms = all_times.max()

    bins = np.arange(cut_start_time, duration_ms + bin_size, bin_size)

    mean_rates = []
    labels = []

    for pop in ordered_pops:
        data = pop_dict[pop]
        spike_times = data[:, 0]

        # 去掉cut_start_time之前的
        mask = spike_times >= cut_start_time
        spike_times = spike_times[mask]

        counts, _ = np.histogram(spike_times, bins=bins)

        rate = counts / pop_sizes[pop] / (bin_size / 1000.0)  # Hz
        smoothed_rate = moving_average(rate, window_size=smooth_window)

        bin_centers = bins[:-1] + bin_size / 2
        ax_rate.plot(bin_centers, smoothed_rate, label=pop)

        mean_rate = np.sum(counts) / pop_sizes[pop] / ((duration_ms - cut_start_time) / 1000.0)
        mean_rates.append(mean_rate)
        labels.append(pop)

    ax_rate.set_title(f"Smoothed Firing Rate Curves for {area}")
    ax_rate.set_xlabel("Time [ms]")
    ax_rate.set_ylabel("Firing rate (Hz)")
    ax_rate.legend()

    # 平均放电率柱状图
    y_pos = np.arange(len(labels))
    ax_bar.barh(y_pos, mean_rates, align="center", color="skyblue")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("Mean firing rate [Hz]")
    ax_bar.set_title("Average firing rates")

    plt.tight_layout()
    plt.savefig(f"{output_folder}/{area}_firing_rate.jpg")
    plt.close(fig_rate)
