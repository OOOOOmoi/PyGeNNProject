import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 CSV 文件
df = pd.read_csv("output/inSyn/V1/S42E4.csv")  # 确保文件路径正确

# 假设你只想画第 0、1、2 号神经元
neurons_to_plot = [0, 1, 2]
time = df.index * 0.1  # 每步0.1ms，转为时间

plt.figure(figsize=(12, 5))
for i in neurons_to_plot:
    plt.plot(time, df.iloc[:, i], label=f'Neuron {i}')

plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.title('Input Currents of Selected Neurons')
plt.legend()
plt.tight_layout()
plt.savefig('output/inSyn/input_currents_S42E4.png', dpi=300)
