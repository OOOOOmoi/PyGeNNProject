import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime
import random
import string

def generate_unique_suffix(length=6):
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"

# 创建输出目录
os.makedirs("raster", exist_ok=True)
os.makedirs("hist", exist_ok=True)

popName = ["H1","E23","S23","P23","V23",
           "E4","S4","P4","V4",
           "E5","S5","P5","V5",
           "E6","S6","P6","V6"]

neurons_per_group = 200
group_spacing = 50
color_map = {
    "H": "purple",
    "E": "red",
    "S": "blue",
    "P": "green",
    "V": "orange"
}

file_list = []
for name in popName:
    filename = f"/home/yangjinhao/PyGenn/SingleColumn/output/spike/{name}_spikes.csv"
    if os.path.exists(filename):
        file_list.append(filename)
    else:
        print(f"Warning: {filename} not found, skipping.")
        file_list.append(None)  # 占位，确保顺序

raster_points = []
y_ticks = []
y_tick_labels = []

current_y_offset = 0
group_avg_rates = []
group_labels = []

for idx, (group_name, file) in enumerate(zip(popName, file_list)):
    y_tick_labels.append(group_name)
    y_ticks.append(current_y_offset + neurons_per_group // 2)
    
    if file is not None:
        df = pd.read_csv(file, comment='#', names=["Time", "NeuronID"])
        df = df[df["Time"] >= 100]  # 省略前100ms数据

        if not df.empty:
            unique_neurons = sorted(df["NeuronID"].unique())
            selected_neurons = unique_neurons[:neurons_per_group]
            filtered_df = df[df["NeuronID"].isin(selected_neurons)]

            neuron_id_map = {nid: i + current_y_offset for i, nid in enumerate(selected_neurons)}
            y_positions = filtered_df["NeuronID"].map(neuron_id_map)
            pop_type = group_name[0]  # 例如 "E23" -> "E"
            color = color_map.get(pop_type, "gray")  # 默认为 gray
            raster_points.append((filtered_df["Time"], y_positions, color))

            duration = 900
            total_neurons = df["NeuronID"].nunique()
            if duration > 0 and total_neurons > 0:
                avg_rate = len(df) / total_neurons / (duration / 1000)
            else:
                avg_rate = 0.0
        else:
            avg_rate = 0.0
    else:
        avg_rate = 0.0  # 文件不存在则速率为 0

    group_avg_rates.append(avg_rate)
    group_labels.append(group_name)
    current_y_offset += neurons_per_group + group_spacing

# 生成文件名后缀
suffix = generate_unique_suffix()

# === Raster Plot ===
plt.figure(figsize=(12, 6))
for times, y_pos, color in raster_points:
    plt.scatter(times, y_pos, s=2, color=color)

plt.yticks(y_ticks, y_tick_labels)
plt.xlabel("Time (ms)")
plt.ylabel("Group")
plt.title(f"Raster Plot (First {neurons_per_group} Neurons per Group)")
plt.tight_layout()
plt.savefig(f"raster/raster_{suffix}.png")

# === Firing Rate Histogram ===
plt.figure(figsize=(10, 4))
plt.bar(group_labels, group_avg_rates, color=[color_map.get(pop[0], "gray") for pop in group_labels])
plt.ylabel("Average Firing Rate (Hz)")
plt.title("Average Firing Rate per Group (Excluding First 100ms)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"hist/histogram_{suffix}.png")
