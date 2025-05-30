import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime
import random
import string
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiColumn import AreaList
def generate_unique_suffix(length=6):
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"

# 创建输出目录
os.makedirs("raster", exist_ok=True)
os.makedirs("hist", exist_ok=True)

# 设置参数
neurons_per_group = 200
group_spacing = 50
color_map = {
    "H": "purple",
    "E": "red",
    "S": "blue",
    "P": "green",
    "V": "orange"
}

popName = ["H1","E23","S23","P23","V23",
           "E4","S4","P4","V4",
           "E5","S5","P5","V5",
           "E6","S6","P6","V6"]
areaName = AreaList

# === 收集文件并按 area 分类 ===
file_list = []
for area in areaName:
    for name in popName:
        filename = f"/home/yangjinhao/PyGenn/MultiColumn/output/spike/{area}_{name}_spikes.csv"
        if os.path.exists(filename):
            file_list.append(filename)
    else:
        print(f"Warning: {filename} not found, skipping.")
        file_list.append(None)  # 占位，确保顺序

area_group_files = defaultdict(list)

for f in file_list:
    if f is None:
        continue
    base = os.path.basename(f).replace('_spikes.csv', '')
    try:
        area, pop = base.split('_', 1)
        area_group_files[area].append((pop, f))
    except ValueError:
        print(f"Skipping invalid file: {f}")

# === 绘制 Raster 和 Histogram ===
suffix = generate_unique_suffix()

# 1. Raster plot
fig_raster, axs_raster = plt.subplots(len(area_group_files), 1, figsize=(12, 5 * len(area_group_files)), sharex=True)

if len(area_group_files) == 1:
    axs_raster = [axs_raster]  # 统一处理

# 2. Histogram
fig_hist, axs_hist = plt.subplots(len(area_group_files), 1, figsize=(10, 4 * len(area_group_files)))

if len(area_group_files) == 1:
    axs_hist = [axs_hist]

# === 主循环每个 area ===
for area_idx, (area, group_files) in enumerate(sorted(area_group_files.items())):
    current_y_offset = 0
    y_ticks = []
    y_labels = []
    raster_points = []
    avg_rates = []
    group_labels = []

    group_file_dict = dict(group_files)
    ordered_group_files = [(pop, group_file_dict[pop]) for pop in popName if pop in group_file_dict]

    for i, (pop, filepath) in enumerate(ordered_group_files):

        try:
            df = pd.read_csv(filepath, comment='#', names=["Time", "NeuronID"])
            df = df[df["Time"] >= 100]  # 忽略前100ms

            if not df.empty:
                # Raster 用部分神经元
                unique_neurons = sorted(df["NeuronID"].unique())
                selected_neurons = unique_neurons[:neurons_per_group]
                filtered_df = df[df["NeuronID"].isin(selected_neurons)]

                neuron_id_map = {nid: idx + current_y_offset for idx, nid in enumerate(selected_neurons)}
                y_positions = filtered_df["NeuronID"].map(neuron_id_map)
                pop_type = pop[0]  # 例如 "E23" -> "E"
                color = color_map.get(pop_type, "gray")  # 默认为 gray
                raster_points.append((filtered_df["Time"], y_positions, color))

                # 计算平均发放率（用所有神经元）
                total_neurons = df["NeuronID"].nunique()
                duration = 900
                if total_neurons > 0 and duration > 0:
                    avg_rate = len(df) / total_neurons / (duration / 1000)
                else:
                    avg_rate = 0.0
            else:
                avg_rate = 0.0
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
            avg_rate = 0.0

        y_ticks.append(current_y_offset + neurons_per_group // 2)
        y_labels.append(pop)
        group_labels.append(pop)
        avg_rates.append(avg_rate)
        current_y_offset += neurons_per_group + group_spacing

    # === 绘制 raster 子图 ===
    ax_raster = axs_raster[area_idx]
    for times, y_pos, color in raster_points:
        ax_raster.scatter(times, y_pos, s=2, color=color)

    ax_raster.set_yticks(y_ticks)
    ax_raster.set_yticklabels(y_labels)
    ax_raster.set_ylabel(f"{area}")
    if area_idx == len(area_group_files) - 1:
        ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_title(f"{area} Raster")

    # === 绘制 histogram 子图 ===
    ax_hist = axs_hist[area_idx]
    ax_hist.bar(group_labels, avg_rates, color=[color_map.get(pop[0], "gray") for pop in group_labels])
    ax_hist.set_ylabel("Avg Firing Rate (Hz)")
    ax_hist.set_title(f"{area} Average Firing Rate")
    ax_hist.set_xticks(range(len(group_labels)))
    ax_hist.set_xticklabels(group_labels, rotation=45)

# === 保存图像 ===
fig_raster.tight_layout()
fig_hist.tight_layout()
fig_raster.savefig(f"raster/raster_{suffix}.png")
fig_hist.savefig(f"hist/hist_{suffix}.png")
