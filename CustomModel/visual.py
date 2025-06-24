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
import argparse
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def generate_unique_suffix(length=3):
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"

def record_spike(neuron_population, spike_data):
    for area, pop_dict in neuron_population.items():
        for pop, p in pop_dict.items():
            spike_times, spike_ids = p.spike_recording_data[0]
            spike_data[area][pop].append(np.column_stack((spike_times, spike_ids)))
    return spike_data

import os
import numpy as np
import glob

def save_spike(spike_data):
    for area, pop_dict in spike_data.items():
        output_dir = f"output/spike/{area}"
        os.makedirs(output_dir, exist_ok=True)

        # 清除该区域下所有旧的 .csv 文件
        for old_file in glob.glob(f"{output_dir}/*.csv"):
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Warning: Failed to delete {old_file}: {e}")

        for pop, data_chunks in pop_dict.items():
            if len(data_chunks) == 0:
                continue  # 避免没有数据时报错
            all_data = np.vstack(data_chunks)
            save_path = f"{output_dir}/{area}_{pop}_spikes.csv"
            np.savetxt(
                save_path,
                all_data,
                delimiter=",",
                fmt=("%f", "%d"),
                header="Times [ms], Neuron ID"
            )

def raster_plot(spike_data, drop=200, neurons_per_group=10, group_spacing=50, model_name=None):
    color_map = {
        "H": "purple",
        "E": "red",
        "S": "blue",
        "P": "green",
        "V": "orange"
    }
    if spike_data==[]:
        print("All spike_data empty, trying to infer from output/spike directory...")
        spike_root = "output/spike"
        if not os.path.exists(spike_root):
            print("No spike output folder found. Exiting raster plot.")
            return

        # 重构 spike_data：从目录结构构建 area 和 pop
        spike_data = {}
        for area in sorted(os.listdir(spike_root)):
            area_path = os.path.join(spike_root, area)
            if not os.path.isdir(area_path):
                continue
            spike_data[area] = {}
            for fname in sorted(os.listdir(area_path)):
                if fname.endswith("_spikes.csv"):
                    pop = fname.replace(f"{area}_", "").replace("_spikes.csv", "")
                    csv_path = os.path.join(area_path, fname)
                    try:
                        loaded_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                        if loaded_data.ndim == 1:
                            loaded_data = loaded_data.reshape(1, -1)
                        spike_data[area][pop] = [loaded_data]
                        print(f"Loaded: {csv_path}")
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")

    fig_raster, axs_raster = plt.subplots(len(spike_data), 1, figsize=(12, 5 * len(spike_data)), sharex=True)
    if len(spike_data) == 1:
        axs_raster = [axs_raster]

    fig_hist, axs_hist = plt.subplots(len(spike_data), 1, figsize=(10, 4 * len(spike_data)))
    if len(spike_data) == 1:
        axs_hist = [axs_hist]
    
    for area_idx, (area, pop_dict) in enumerate(spike_data.items()):
        current_y_offset = 0
        raster_point=[]
        avg_rates = []
        y_ticks = []
        y_labels = []
        group_labels = []
        for pop, data_chunks in pop_dict.items():
            all_spikes = np.vstack(data_chunks)
            times = all_spikes[:, 0]
            ids = all_spikes[:, 1].astype(int)
            mask = times >= drop
            times = times[mask]
            ids = ids[mask]
            if ids.size > 0:
                unique_neurons = np.unique(ids)
                selected_neurons = unique_neurons[:neurons_per_group]
                mask = np.isin(ids, selected_neurons)
                filtered_times = times[mask]
                filtered_ids = ids[mask]

                neuron_id_map = {nid: idx + current_y_offset for idx, nid in enumerate(selected_neurons)}
                y_positions = np.array([neuron_id_map[i] for i in filtered_ids])
                pop_type = pop[0]
                color = color_map.get(pop_type, "gray")
                raster_point.append((filtered_times, y_positions, color))

                duration = times.max() - drop
                total_neurons = np.unique(ids).size
                avg_rate = times.size / total_neurons / (duration / 1000) if total_neurons > 0 and duration > 0 else 0.0
            else:
                avg_rate = 0.0
            avg_rates.append(avg_rate)
            y_ticks.append(current_y_offset + neurons_per_group // 2)
            y_labels.append(pop)
            group_labels.append(pop)
            current_y_offset += neurons_per_group + group_spacing
        ax_raster=axs_raster[area_idx]
        for times, y_pos, color in raster_point:
            ax_raster.scatter(times, y_pos, s=2, color=color)

        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_labels)
        ax_raster.set_ylabel(f"{area}")
        if area_idx == len(spike_data) - 1:
            ax_raster.set_xlabel("Time (ms)")
        if model_name:
            ax_raster.set_title(f"{area} Raster - {model_name}")
        else:
            ax_raster.set_title(f"{area} Raster")

        ax_hist = axs_hist[area_idx]
        bars = ax_hist.bar(group_labels, avg_rates, color=[color_map.get(pop[0], "gray") for pop in group_labels])
        ax_hist.set_ylabel("Avg Firing Rate (Hz)")

        # 添加数值标签在每个柱子上方
        for bar, rate in zip(bars, avg_rates):
            height = bar.get_height()
            ax_hist.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f"{rate:.1f}",
                        ha='center', va='bottom', fontsize=8)
        if model_name:
            ax_hist.set_title(f"{area} Rate - {model_name}")
        else:
            ax_hist.set_title(f"{area} Average Firing Rate")
        ax_hist.set_xticks(range(len(group_labels)))
        ax_hist.set_xticklabels(group_labels, rotation=45)

    fig_raster.tight_layout()
    fig_hist.tight_layout()
    os.makedirs("output/raster", exist_ok=True)
    os.makedirs("output/hist", exist_ok=True)
    suffix = generate_unique_suffix()
    fig_raster.savefig(f"output/raster/raster_{suffix}.png")
    fig_hist.savefig(f"output/hist/hist_{suffix}.png")