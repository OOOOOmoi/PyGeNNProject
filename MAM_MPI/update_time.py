import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取日志文件
# =========================
log_file = "output.txt"

records = []

# 示例匹配行：
# [Worker 4] timestep=222 step_time=95.565 ms update_V=64.596 ms update_inSyn=204.331 ms update_decay=25.900 ms update_spike=0.140 ms total_update=390.535 ms

pattern = re.compile(
    r"\[Worker (\d+)\]\s+"
    r"timestep=(\d+)\s+"
    r"step_time=([\d.]+) ms\s+"
    r"update_V=([\d.]+) ms\s+"
    r"update_inSyn=([\d.]+) ms\s+"
    r"update_decay=([\d.]+) ms\s+"
    r"update_spike=([\d.]+) ms\s+"
    r"total_update=([\d.]+) ms"
)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            records.append({
                "worker": int(match.group(1)),
                "timestep": int(match.group(2)),
                "step_time": float(match.group(3)),
                "update_V": float(match.group(4)),
                "update_inSyn": float(match.group(5)),
                "update_decay": float(match.group(6)),
                "update_spike": float(match.group(7)),
                "total_update": float(match.group(8)),
            })

# =========================
# 2. 转成 DataFrame
# =========================
df = pd.DataFrame(records)
df = df.sort_values(by=["worker", "timestep"])

print("解析完成，数据样例：")
print(df.head())

# 如果你想保存成 CSV：
df.to_csv("worker_timestep_stats.csv", index=False, encoding="utf-8-sig")

# =========================
# 3. 指定要分析的 Worker
# =========================
worker_id = 25

df_w = df[df["worker"] == worker_id]
df_w = df_w[df_w["timestep"] >= 10]
# =========================
# 4. 画随 timestep 变化的耗时曲线
# =========================
plt.figure(figsize=(12, 8))

plt.plot(df_w["timestep"], df_w["update_V"], label="update_V")
plt.plot(df_w["timestep"], df_w["update_inSyn"], label="update_inSyn")
plt.plot(df_w["timestep"], df_w["update_decay"], label="update_decay")
plt.plot(df_w["timestep"], df_w["update_spike"], label="update_spike")
plt.plot(df_w["timestep"], df_w["step_time"], label="step_time", linestyle="--")
plt.plot(df_w["timestep"], df_w["total_update"], label="total_update", linewidth=2)

plt.xlabel("Timestep")
plt.ylabel("Time (ms)")
plt.title(f"Worker {worker_id} Time Analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"time_analysis/worker_{worker_id}_time_analysis_final.png", dpi=300)

