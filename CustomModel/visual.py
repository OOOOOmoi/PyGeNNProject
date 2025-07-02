import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime
import random
import string
from collections import defaultdict
import sys
from scipy.ndimage import gaussian_filter1d
from psd import plot_psd
from rate_curve import plot_firing_rate_curve
from config import connection_params
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def smooth_firing_rate(spike_times, total_neurons, sample_bin=1, sigma=3, drop=200):
    t_min = drop
    t_max = int(spike_times.max()) + 1
    time_bins = np.arange(t_min, t_max + 1, sample_bin)
    binned_rate, _ = np.histogram(spike_times, bins=time_bins)
    binned_rate = binned_rate * 1000 / total_neurons
    return gaussian_filter1d(binned_rate.astype(float), sigma=sigma), time_bins

def generate_unique_suffix(length=3):
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"

def visualize(spike_data, duration=1000, drop=200, neurons_per_group=200, group_spacing=50, 
                model_name=None, NeuronNumber=None, sample_bin=1, vis_content=None):
    if vis_content is None:
        vis_content = set()
    color_map = {
        "H": "purple",
        "E": "red",
        "S": "blue",
        "P": "green",
        "V": "orange"
    }
    suffix = generate_unique_suffix()
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
    
    input=connection_params['input']

    for area_idx, (area, pop_dict) in enumerate(spike_data.items()):
        current_y_offset = 0
        raster_point=[]
        avg_rates = []
        y_ticks = []
        y_labels = []
        group_labels = []
        all_spike = []
        layer_spikes_dict = defaultdict(list)
        for pop, data_chunks in pop_dict.items():
            all_spikes = np.vstack(data_chunks)
            times = all_spikes[:, 0]
            ids = all_spikes[:, 1].astype(int)
            mask = times >= drop
            times = times[mask]
            ids = ids[mask]
            total_neurons = NeuronNumber[area][pop]
            selected_neurons = np.random.choice(total_neurons, neurons_per_group, replace=False)
            
            # 筛选当前 selected 神经元的放电
            mask = np.isin(ids, selected_neurons)
            filtered_times = times[mask]
            filtered_ids = ids[mask]

            if filtered_ids.size == 0:
                avg_rate = 0.0
                avg_rates.append(avg_rate)
                y_ticks.append(current_y_offset + neurons_per_group // 2)
                y_labels.append(pop+"_"+str(input[pop]))
                group_labels.append(pop)
                current_y_offset += neurons_per_group + group_spacing
                continue

            # 构造 raster y 位置（哪怕有些 neuron 没放电也不会出错）
            neuron_id_map = {nid: idx + current_y_offset for idx, nid in enumerate(selected_neurons)}
            y_positions = np.array([neuron_id_map[i] for i in filtered_ids])

            pop_type = pop[0]
            color = color_map.get(pop_type, "gray")
            raster_point.append((filtered_times, y_positions, color))

            duration_ms = duration - drop
            n_selected = len(selected_neurons)
            avg_rate = len(times) / total_neurons / (duration_ms / 1000) if n_selected > 0 and duration_ms > 0 else 0.0


            layer_id = ''.join(filter(str.isdigit, pop))
            if layer_id:
                if layer_id not in layer_spikes_dict:
                    layer_spikes_dict[layer_id] = {"spike_times": [], "neuron_count": 0}
                layer_spikes_dict[layer_id]["spike_times"].extend(filtered_times)

                # 从 NeuronNumber[area][pop] 读取神经元数量并累加
                if area in NeuronNumber and pop in NeuronNumber[area]:
                    layer_spikes_dict[layer_id]["neuron_count"] += NeuronNumber[area][pop]
                else:
                    print(f"Warning: neuron count not found for area {area}, pop {pop}")
            
            all_spike.append(filtered_times)
            smoothed_rate, time_bins = smooth_firing_rate(filtered_times, total_neurons, sample_bin=sample_bin)
            if 'pop-psd' in vis_content:
                plot_psd(smoothed_rate, time_bins, model_name, sample_bin, 
                        suffix, area, layer=None, pop=pop, drop=drop)
            if 'pop-rate' in vis_content:
                plot_firing_rate_curve(smoothed_rate, time_bins, suffix, model_name, 
                                    area=area, layer=None, pop=pop)
            avg_rates.append(avg_rate)
            y_ticks.append(current_y_offset + neurons_per_group // 2)
            y_labels.append(pop+"_"+str(input[pop]))
            group_labels.append(pop)
            current_y_offset += neurons_per_group + group_spacing
        if 'layer-psd' in vis_content or 'layer-rate' in vis_content:
            for layer, layer_data in layer_spikes_dict.items():
                spikes = layer_data["spike_times"]
                neuron_count = layer_data["neuron_count"]
                if len(spikes) < 10:
                    continue
                spike_times = np.array(spikes)
                smoothed_rate, time_bins = smooth_firing_rate(spike_times, neuron_count, sample_bin=sample_bin)
                if 'layer-psd' in vis_content:
                    plot_psd(smoothed_rate, time_bins, model_name, sample_bin, 
                            suffix, area, layer=layer, pop=None, drop=drop)
                if 'layer-rate' in vis_content:
                    plot_firing_rate_curve(smoothed_rate, time_bins, suffix, model_name, 
                                        area=area, layer=layer, pop=None)

        if 'area-psd' in vis_content or 'area-rate' in vis_content:
            all_spike = np.concatenate(all_spike) if all_spike else np.array([])
            smoothed_rate, time_bins = smooth_firing_rate(all_spike, NeuronNumber[area]['total'], sample_bin=sample_bin)
            if 'area-psd' in vis_content:
                plot_psd(smoothed_rate, time_bins, model_name, sample_bin, 
                        suffix, area, layer=None, pop=None)
            if 'area-rate' in vis_content:
                plot_firing_rate_curve(smoothed_rate, time_bins, suffix, model_name, 
                                    area=area, layer=None, pop=None)

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
    fig_raster.savefig(f"output/raster/raster_{suffix}.png")
    fig_hist.savefig(f"output/hist/hist_{suffix}.png")