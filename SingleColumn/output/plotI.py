import numpy as np
import matplotlib.pyplot as plt

# 加载数据（每一列代表一个神经元）
src = "E23"
tar = "E23"
data = np.loadtxt(f"inSyn/{tar}/{src}2{tar}.csv", delimiter=",")


start_step = int(200 / 0.1)

# 创建时间轴（单位：ms）
time = np.arange(data.shape[0]) * 0.1
time = time[start_step:]  # 省略前 200ms
data = data[start_step:]
# 设置绘图
plt.figure(figsize=(10, 6))

# 绘制前 N 个神经元的曲线，避免图太挤
N = min(2, data.shape[1])  # 只画前10个神经元（可按需修改）
for i in range(N):
    plt.plot(time, data[:, i], label=f'Neuron {i}')

plt.xlabel("Time")
plt.ylabel("Input Current")
plt.title("Synaptic Current Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"inSyn/{src}2{tar}.png")
