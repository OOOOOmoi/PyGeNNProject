import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from visual import smooth_firing_rate

def visualize_single(suffix, spike_data, duration=1000, drop=200, neurons_per_group=200, group_spacing=50, 
                model_name=None, NeuronNumber=None, sample_bin=1, vis_content=None):
    if vis_content is None:
        vis_content = set()
    color_map = {
        "H": "purple",
        "E": "red",
        "I": "blue",
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

    fig_raster, axs_raster = plt.subplots(2, figsize=(20, 10), sharex=True)

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
                y_labels.append(pop)
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

            all_spike.append(filtered_times)
            smoothed_rate, time_bins = smooth_firing_rate(times, total_neurons, sample_bin=sample_bin, drop=drop)
            smoothed_rate_normal = smoothed_rate / max(smoothed_rate)
            time_axis = (time_bins[:-1] + time_bins[1:]) / 2
            axs_raster[1].plot(time_axis, smoothed_rate_normal, label=pop, color=color)


            avg_rates.append(avg_rate)
            y_ticks.append(current_y_offset + neurons_per_group // 2)
            y_labels.append(pop)
            group_labels.append(pop)
            current_y_offset += neurons_per_group + group_spacing
        ax_raster=axs_raster
        for times, y_pos, color in raster_point:
            ax_raster[0].scatter(times, y_pos, s=1, color=color)

        ax_raster[0].set_yticks(y_ticks)
        ax_raster[0].set_yticklabels(y_labels)
        ax_raster[0].set_ylabel(f"{area}")
        ax_raster[1].set_xlabel("Time (ms)")
        ax_raster[1].set_ylabel("Normalized firing rate")
        ax_raster[1].legend()
        if model_name:
            ax_raster[0].set_title(f"{area} Raster - {model_name}")
        else:
            ax_raster[0].set_title(f"{area} Raster")

    fig_raster.tight_layout()
    os.makedirs("output/raster", exist_ok=True)
    fig_raster.savefig(f"output/raster/raster_{model_name}.png")