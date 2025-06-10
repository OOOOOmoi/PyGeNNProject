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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiColumn import AreaList, NeuronNumber

parser = argparse.ArgumentParser()
parser.add_argument("--use-model-file", action="store_true", help="是否使用 last_model_name.txt 中的模型名称")
parser.add_argument("--drop", type=int, default=200, help="drop out time")
parser.add_argument("--pop-psd", action="store_true", help="wether plot psd")
parser.add_argument("--layer-psd", action="store_true", help="wether plot layer specific psd")
parser.add_argument("--time-bins", type=float, default=1, help="length of rate bin")
args = parser.parse_args()

def truncate_psd(f, psd, f_max=100):
    if f is None or psd is None:
        return None, None
    mask = f <= f_max
    return f[mask], psd[mask]

if args.use_model_file:
    with open("last_model_name.txt", "r") as f:
        model_name = f.read().strip()
else:
    model_name = None

def generate_unique_suffix(length=6):
    date_str = datetime.datetime.now().strftime("%m%d-%H%M")
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{date_str}_{rand_str}"

os.makedirs("raster", exist_ok=True)
os.makedirs("hist", exist_ok=True)
os.makedirs("psd", exist_ok=True)

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

file_list = []
for area in areaName:
    for name in popName:
        filename = f"/home/yangjinhao/PyGenn/MultiColumn/output/spike/{area}_{name}_spikes.csv"
        if os.path.exists(filename):
            file_list.append(filename)
        else:
            print(f"Warning: {filename} not found, skipping.")
            file_list.append(None)

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

suffix = generate_unique_suffix()

fig_raster, axs_raster = plt.subplots(len(area_group_files), 1, figsize=(12, 5 * len(area_group_files)), sharex=True)
if len(area_group_files) == 1:
    axs_raster = [axs_raster]

fig_hist, axs_hist = plt.subplots(len(area_group_files), 1, figsize=(10, 4 * len(area_group_files)))
if len(area_group_files) == 1:
    axs_hist = [axs_hist]

for area_idx, (area, group_files) in enumerate(sorted(area_group_files.items())):
    current_y_offset = 0
    y_ticks = []
    y_labels = []
    raster_points = []
    avg_rates = []
    group_labels = []

    group_file_dict = dict(group_files)
    ordered_group_files = [(pop, group_file_dict[pop]) for pop in popName if pop in group_file_dict]

    layer_spikes_dict = defaultdict(list)

    for i, (pop, filepath) in enumerate(ordered_group_files):
        try:
            df = pd.read_csv(filepath, comment='#', names=["Time", "NeuronID"])
            df = df[df["Time"] >= args.drop]

            if not df.empty:
                unique_neurons = sorted(df["NeuronID"].unique())
                selected_neurons = unique_neurons[:neurons_per_group]
                filtered_df = df[df["NeuronID"].isin(selected_neurons)]

                neuron_id_map = {nid: idx + current_y_offset for idx, nid in enumerate(selected_neurons)}
                y_positions = filtered_df["NeuronID"].map(neuron_id_map)
                pop_type = pop[0]
                color = color_map.get(pop_type, "gray")
                raster_points.append((filtered_df["Time"], y_positions, color))

                duration = df["Time"].max() - args.drop
                total_neurons = df["NeuronID"].nunique()
                avg_rate = len(df) / total_neurons / (duration / 1000) if total_neurons > 0 and duration > 0 else 0.0

                layer_id = ''.join(filter(str.isdigit, pop))
                if layer_id:
                    if layer_id not in layer_spikes_dict:
                        layer_spikes_dict[layer_id] = {"spike_times": [], "neuron_count": 0}
                    layer_spikes_dict[layer_id]["spike_times"].extend(df["Time"].values)

                    # 从 NeuronNumber[area][pop] 读取神经元数量并累加
                    if area in NeuronNumber and pop in NeuronNumber[area]:
                        layer_spikes_dict[layer_id]["neuron_count"] += NeuronNumber[area][pop]
                    else:
                        print(f"Warning: neuron count not found for area {area}, pop {pop}")
                
                # === PSD 分析（群体 firing rate） ===
                if args.pop_psd:
                    spike_times = df["Time"].values
                    t_min = args.drop
                    t_max = int(df["Time"].max()) + 1
                    time_bins = np.arange(t_min, t_max + 1, args.time_bins)
                    binned_rate, _ = np.histogram(spike_times, bins=time_bins)
                    binned_rate = binned_rate * 1000 / df["NeuronID"].max()
                    rate_smoothed = gaussian_filter1d(binned_rate.astype(float), sigma=3)

                    stim_start = stim_end = None
                    if model_name and "_stim" in model_name:
                        try:
                            stim_start = float(model_name.split("_start")[1].split("s")[0]) * 1000
                            stim_end = float(model_name.split("_end")[1].split("s")[0]) * 1000
                        except Exception as e:
                            print(f"Error parsing stim time: {e}")

                    fs = 1 / args.time_bins * 1000
                    if stim_start and stim_end:
                        t_range = time_bins[:-1]
                        stim_mask = (t_range >= stim_start) & (t_range < stim_end)
                        nostim_mask = ~stim_mask

                        if stim_mask.sum() > 10:
                            f_stim, psd_stim = welch(rate_smoothed[stim_mask], fs=fs, nperseg=256)
                        else:
                            f_stim, psd_stim = None, None

                        if nostim_mask.sum() > 10:
                            f_nostim, psd_nostim = welch(rate_smoothed[nostim_mask], fs=fs, nperseg=256)
                        else:
                            f_nostim, psd_nostim = None, None
                    else:
                        f_nostim, psd_nostim = welch(rate_smoothed, fs=fs, nperseg=256)
                        f_stim, psd_stim = None, None

                    f_nostim, psd_nostim = truncate_psd(f_nostim, psd_nostim)
                    f_stim, psd_stim = truncate_psd(f_stim, psd_stim)

                    fig_psd, axs_psd = plt.subplots(1, 2 if psd_stim is not None else 1, figsize=(10, 4))
                    if psd_stim is not None:
                        axs_psd[0].plot(f_nostim, psd_nostim, color="black", label="No Stim")
                        axs_psd[0].set_title("No Stim")
                        axs_psd[1].plot(f_stim, psd_stim, color="orange", label="Stim")
                        axs_psd[1].set_title("Stim")
                        for ax in axs_psd:
                            ax.set_xlabel("Frequency (Hz)")
                            ax.set_ylabel("Power")
                            ax.grid(True)
                    else:
                        axs_psd.plot(f_nostim, psd_nostim, color="black", label="No Stim")
                        axs_psd.set_xlabel("Frequency (Hz)")
                        axs_psd.set_ylabel("Power")
                        axs_psd.set_title("No Stim")
                        axs_psd.grid(True)

                    fig_psd.suptitle(f"{area} - {pop} Power Spectrum")
                    fig_psd.tight_layout()
                    os.makedirs(f"psd/{area}", exist_ok=True)
                    fig_psd.savefig(f"psd/{area}/{pop}_psd.png")
                    plt.close(fig_psd)

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

    if args.layer_psd:
        for layer, layer_data in layer_spikes_dict.items():
            spikes = layer_data["spike_times"]
            neuron_count = layer_data["neuron_count"]
            if len(spikes) < 10:
                continue
            spike_times = np.array(spikes)
            t_min = args.drop
            t_max = int(spike_times.max()) + 1
            time_bins = np.arange(t_min, t_max + 1, args.time_bins)
            binned_rate, _ = np.histogram(spike_times, bins=time_bins)
            binned_rate = binned_rate * 1000 / df["NeuronID"].max()
            rate_smoothed = gaussian_filter1d(binned_rate.astype(float), sigma=3)

            stim_start = stim_end = None
            if model_name and "_stim" in model_name:
                try:
                    stim_start = float(model_name.split("_start")[1].split("s")[0]) * 1000
                    stim_end = float(model_name.split("_end")[1].split("s")[0]) * 1000
                except Exception as e:
                    print(f"Error parsing stim time: {e}")
            fs = 1 / args.time_bins * 1000
            if stim_start and stim_end:
                t_range = time_bins[:-1]
                stim_mask = (t_range >= stim_start) & (t_range < stim_end)
                nostim_mask = ~stim_mask
                f_stim, psd_stim = welch(rate_smoothed[stim_mask], fs=fs, nperseg=256) if stim_mask.sum() > 10 else (None, None)
                f_nostim, psd_nostim = welch(rate_smoothed[nostim_mask], fs=fs, nperseg=256) if nostim_mask.sum() > 10 else (None, None)
            else:
                f_nostim, psd_nostim = welch(rate_smoothed, fs=fs, nperseg=256)
                f_stim, psd_stim = None, None

            f_nostim, psd_nostim = truncate_psd(f_nostim, psd_nostim)
            f_stim, psd_stim = truncate_psd(f_stim, psd_stim)

            fig_layer_psd, axs_layer_psd = plt.subplots(1, 2 if psd_stim is not None else 1, figsize=(10, 4))
            if psd_stim is not None:
                axs_layer_psd[0].plot(f_nostim, psd_nostim, color="black")
                axs_layer_psd[0].set_title("No Stim")
                axs_layer_psd[1].plot(f_stim, psd_stim, color="orange")
                axs_layer_psd[1].set_title("Stim")
                for ax in axs_layer_psd:
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Power")
                    ax.grid(True)
            else:
                axs_layer_psd.plot(f_nostim, psd_nostim, color="black")
                axs_layer_psd.set_xlabel("Frequency (Hz)")
                axs_layer_psd.set_ylabel("Power")
                axs_layer_psd.set_title("No Stim")
                axs_layer_psd.grid(True)

            fig_layer_psd.suptitle(f"{area} - Layer {layer} Power Spectrum")
            fig_layer_psd.tight_layout()
            os.makedirs(f"psd/{area}", exist_ok=True)
            fig_layer_psd.savefig(f"psd/{area}/{layer}_psd.png")
            plt.close(fig_layer_psd)


    ax_raster = axs_raster[area_idx]
    for times, y_pos, color in raster_points:
        ax_raster.scatter(times, y_pos, s=2, color=color)

    ax_raster.set_yticks(y_ticks)
    ax_raster.set_yticklabels(y_labels)
    ax_raster.set_ylabel(f"{area}")
    if area_idx == len(area_group_files) - 1:
        ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_title(f"{area} Raster")

    ax_hist = axs_hist[area_idx]
    ax_hist.bar(group_labels, avg_rates, color=[color_map.get(pop[0], "gray") for pop in group_labels])
    ax_hist.set_ylabel("Avg Firing Rate (Hz)")
    ax_hist.set_title(f"{area} Average Firing Rate")
    ax_hist.set_xticks(range(len(group_labels)))
    ax_hist.set_xticklabels(group_labels, rotation=45)

fig_raster.tight_layout()
fig_hist.tight_layout()

if model_name:
    fig_raster.savefig(f"raster/raster_{model_name}.png")
    fig_hist.savefig(f"hist/hist_{model_name}.png")
else:
    fig_raster.savefig(f"raster/raster_{suffix}.png")
    fig_hist.savefig(f"hist/hist_{suffix}.png")