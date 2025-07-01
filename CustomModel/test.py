import pandas as pd

# 加载 CSV
df = pd.read_csv("/home/yangjinhao/PyGenn/CustomModel/output/spike/V1/V1_S4_spikes.csv", comment='#', header=None, names=["Time", "NeuronID"])
df = df[df["Time"] >= 200]
# 总放电次数
total_spikes = len(df)

# 时间范围（转换成秒）
duration_ms = df["Time"].max() - df["Time"].min()
duration_s = duration_ms / 1000

# 神经元数量（唯一 ID 数）
num_neurons = df["NeuronID"].nunique()

# 平均放电率 = 总放电数 / 神经元数 / 时间（秒）
if num_neurons > 0 and duration_s > 0:
    avg_rate = total_spikes / num_neurons / duration_s
else:
    avg_rate = 0.0

print(f"平均放电率: {avg_rate:.2f} Hz")
