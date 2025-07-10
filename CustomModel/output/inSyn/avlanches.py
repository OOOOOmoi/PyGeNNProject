import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 读取 CSV 数据 ===
filename = "V1/V4/S4_2_V4.csv"  # 替换为你的文件名
data = pd.read_csv(filename, header=None).values  # shape: (timesteps, neurons)
data = data[10000:, :]  # 删除前10000行
num_timesteps, num_neurons = data.shape

avalanche_sizes = []

for neuron in range(num_neurons):
    current = data[:, neuron]

    # 检查 current 是否为负，表示电流输入（活跃期）
    is_nonzero = current < 0
    d = np.diff(np.concatenate([[0], is_nonzero.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1

    for start, end in zip(starts, ends):
        seg = current[start:end + 1]
        dI = -np.diff(seg)  # 电流上升表示可能是spike

        spike_counts = 0
        for delta in dI:
            if delta > 0.1:
                spike_counts += int(round(delta / 0.2))  # 估算spike数量

        if spike_counts > 0:
            avalanche_sizes.append(spike_counts)

# === 构建直方图 ===
if avalanche_sizes:
    avalanche_sizes = np.array(avalanche_sizes)
    edges = np.arange(1, avalanche_sizes.max() + 2)
    counts, _ = np.histogram(avalanche_sizes, bins=edges)
    bin_centers = edges[:-1] + 0.5
    pdf_values = counts / counts.sum()

    # === 绘图 ===
    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, pdf_values, '-o', linewidth=1.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Avalanche Size (spike count)')
    plt.ylabel('Probability Density')
    plt.title('Avalanche Size Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avalanche_distribution.png")
    plt.show()
else:
    print("No avalanches detected.")
