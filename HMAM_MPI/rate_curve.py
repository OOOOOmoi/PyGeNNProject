import os
import matplotlib.pyplot as plt
import numpy as np

def plot_firing_rate_curve(rate_smoothed, time_bins, suffix, model_name, 
                           area, layer=None, pop=None):
    os.makedirs("output/rate", exist_ok=True)
    
    # 构造时间轴
    time_axis = (time_bins[:-1] + time_bins[1:]) / 2  # ms

    # 创建图像
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, rate_smoothed, color='blue')
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (Hz)")
    plt.grid(True)
    plt.tight_layout()

    # 设置输出路径
    if layer is None and pop is not None:
        title = f"{area} - {pop} Firing Rate"
        if model_name:
            title += f" - {model_name}"
        plt.title(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_dir = f"output/rate/{area}/pop"
        file_prefix = f"{out_dir}/{pop}"
    elif layer is not None and pop is None:
        title = f"{area} - Layer {layer} Firing Rate"
        if model_name:
            title += f" - {model_name}"
        plt.title(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_dir = f"output/rate/{area}/layer"
        file_prefix = f"{out_dir}/{layer}"
    else:
        title = f"{area} Firing Rate"
        if model_name:
            title += f" - {model_name}"
        plt.title(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_dir = f"output/rate/{area}"
        file_prefix = f"{out_dir}/firing_rate"

    # 创建目录
    os.makedirs(out_dir, exist_ok=True)

    # 保存图像
    plt.savefig(f"{file_prefix}.png")
    plt.close()

    # 保存数据为 CSV
    csv_data = np.column_stack((time_axis, rate_smoothed))
    np.savetxt(f"{file_prefix}.csv", csv_data, delimiter=",", header="Time(ms),FiringRate(Hz)", comments='', fmt="%.6f")
