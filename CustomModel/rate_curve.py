import os
import matplotlib.pyplot as plt

def plot_firing_rate_curve(rate_smoothed, time_bins, suffix, model_name, 
                           area, layer=None, pop=None):
    
    # 构造时间轴：用每个 bin 的中心时间点表示
    time_axis = (time_bins[:-1] + time_bins[1:]) / 2  # ms

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, rate_smoothed, color='blue')
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (Hz)")
    plt.grid(True)

    plt.tight_layout()
    if layer is None and pop is not None:
        plt.title(f"{area} - {pop} Firing Rate - {model_name}" if model_name else f"{area} - {pop} Firing Rate")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = f"output/rate/{area}/pop/{pop}_{suffix}.png"
        os.makedirs(f"output/rate/{area}/pop", exist_ok=True)
        plt.savefig(save_path)
    elif layer is not None and pop is None:
        plt.title(f"{area} - Layer {layer} Firing Rate - {model_name}" if model_name else f"{area} - Layer {layer} Firing Rate")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = f"output/rate/{area}/layer/{layer}_{suffix}.png"
        os.makedirs(f"output/rate/{area}/layer", exist_ok=True)
        plt.savefig(save_path)
    elif layer is None and pop is None:
        plt.title(f"{area} Firing Rate - {model_name}" if model_name else f"{area} Firing Rate")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = f"output/rate/{area}/firing_rate_{suffix}.png"
        os.makedirs(f"output/rate/{area}/", exist_ok=True)
        plt.savefig(save_path)
    plt.close()