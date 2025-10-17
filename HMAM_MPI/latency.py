import re
import pandas as pd
import matplotlib.pyplot as plt
# 读取文件
with open("./logs/out/1/rank.00/stdout", "r") as f:
    lines = f.readlines()

# 2️⃣ 提取信息
pattern = re.compile(
    r"\[step (\d+)\].*from worker (\d+) - latency ([\d\.]+) ms, .* sim_step_time ([\d\.]+) ms"
)

records = []
for line in lines:
    m = pattern.search(line)
    if m:
        step = int(m.group(1))
        worker = int(m.group(2))
        latency = float(m.group(3))
        sim_time = float(m.group(4))
        records.append((step, worker, latency, sim_time))

df = pd.DataFrame(records, columns=["step", "worker", "latency_ms", "sim_step_time_ms"])

# 3️⃣ 去掉前1000个时间步
df = df[df["step"] > 1000].reset_index(drop=True)

print(f"✅ 数据读取完成：剩余 {len(df)} 行, {df['step'].nunique()} 个时间步, {df['worker'].nunique()} 个 worker")

# 4️⃣ 绘制分布图（Histogram + KDE）
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["latency_ms"]*10, bins=500, alpha=0.7, edgecolor='black')
plt.title("Latency")
plt.xlabel("Latency (ms)")
plt.ylabel("frequency")

plt.subplot(1, 2, 2)
plt.hist(df["sim_step_time_ms"]*10, bins=500, alpha=0.7, edgecolor='black', color='orange')
plt.title("Simulation Step Time")
plt.xlabel("Step Time (ms)")
plt.ylabel("frequency")

plt.tight_layout()
plt.savefig("latency_simtime_distribution.png", dpi=300)