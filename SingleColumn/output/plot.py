import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime
import random
import string

def generate_unique_suffix(length=6):
    # 当前日期（例如：2025-05-20）
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    # 随机字符串（大写+数字）
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"


# === 配置参数 ===
neurons_per_group = 200  # 每组显示多少个神经元
group_spacing = 5       # 群体之间纵向间隔
colors = ['red', 'blue']

# === 文件读取 ===
file_list = sorted(glob.glob("*_spikes.csv"))

raster_points = []
y_ticks = []
y_tick_labels = []

current_y_offset = 0
group_avg_rates = []
group_labels = []

for idx, file in enumerate(file_list):
    group_name = os.path.basename(file).split('_spikes.csv')[0]
    df = pd.read_csv(file, comment='#', names=["Time", "NeuronID"])

    unique_neurons = sorted(df["NeuronID"].unique())
    selected_neurons = unique_neurons[:neurons_per_group]
    filtered_df = df[df["NeuronID"].isin(selected_neurons)]

    # 重新映射 Neuron ID 到连续的 Y 坐标
    neuron_id_map = {nid: i + current_y_offset for i, nid in enumerate(selected_neurons)}
    y_positions = filtered_df["NeuronID"].map(neuron_id_map)

    raster_points.append((filtered_df["Time"], y_positions, colors[idx % 2]))

    # Y 轴标签在中间
    middle_y = current_y_offset + neurons_per_group // 2
    y_ticks.append(middle_y)
    y_tick_labels.append(group_name)

    current_y_offset += neurons_per_group + group_spacing

    # 平均发放率（按被选中的 neuron）
    duration = df["Time"].max() - df["Time"].min()
    avg_rate = len(filtered_df) / neurons_per_group / (duration / 1000)
    group_avg_rates.append(avg_rate)
    group_labels.append(group_name)


suffix = generate_unique_suffix()
# === 绘制 Raster Plot ===
plt.figure(figsize=(12, 6))
for times, y_pos, color in raster_points:
    plt.scatter(times, y_pos, s=2, color=color)

plt.yticks(y_ticks, y_tick_labels)
plt.xlabel("Time (ms)")
plt.ylabel("Group")
plt.title(f"Raster Plot (First {neurons_per_group} Neurons per Group)")
plt.tight_layout()
plt.savefig(f"raster/raster_{suffix}.png", dpi=300)

# === 绘制 发放率直方图 ===
plt.figure(figsize=(10, 4))
plt.bar(group_labels, group_avg_rates, color=[colors[i % 2] for i in range(len(group_labels))])
plt.ylabel("Average Firing Rate (Hz)")
plt.title("Average Firing Rate per Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"hist/histogram_{suffix}.png", dpi=300)
