import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import get_NN, remove_dash_from_index_columns, net, layer_map
from collections import defaultdict
import os
area_list = net["area_list"]
area_list = [s.replace("-", "") for s in area_list]
layer_list = net["layer_list"]
pop_list = net["population_list"]
NN=get_NN()
NN = remove_dash_from_index_columns(NN)
neuronNumber = defaultdict(dict)
for area in area_list:
    for layer in layer_list:
        for pop in pop_list:
            if (area, layer, pop) in NN.index:
                popNum = NN.loc[(area, layer, pop)]
                neuronNumber[area][pop+layer_map[layer]] = popNum


# =========================
# 用户需要配置的部分
# =========================

VERSIONS = {
    "sGPU": "/home/yangjinhao/PyGenn/HMAM/output",
    "mGPU": "/home/yangjinhao/PyGenn/HMAM_MPI",
}

SPIKE_SUBDIR = os.path.join("spike")


SIMULATION_TIME_MS = 500.0   # 如果你知道总仿真时间（ms）
SIMULATION_TIME_S = SIMULATION_TIME_MS / 1000.0


# =========================
# 工具函数
# =========================

def parse_population_name(filename, area):
    """
    从文件名中解析 population 名称
    area_pop_spikes.csv -> pop
    """
    name = filename.replace(".csv", "")
    prefix = f"{area}_"
    suffix = "_spikes"
    return name.replace(prefix, "").replace(suffix, "")


def compute_population_firing_rate(csv_file, neuron_num, T):
    """
    群体平均 firing rate (Hz)
    """
    # 只数行数即可
    with open(csv_file, "r") as f:
        n_spikes = sum(
            1 for line in f
            if line.strip() and not line.startswith("#")
        )

    return n_spikes / (neuron_num * T)


# =========================
# 主逻辑：遍历目录 + 汇总数据
# =========================

def collect_firing_rate_data():
    """
    返回一个 long-format DataFrame:
    version | area | population | neuron_id | firing_rate
    """
    records = []

    for version, base_folder in VERSIONS.items():
        spike_root = os.path.join(base_folder, SPIKE_SUBDIR)

        for area in os.listdir(spike_root):
            area_path = os.path.join(spike_root, area)
            if not os.path.isdir(area_path):
                continue

            for fname in os.listdir(area_path):
                if not fname.endswith(".csv"):
                    continue

                pop = parse_population_name(fname, area)

                if area not in neuronNumber or pop not in neuronNumber[area]:
                    raise ValueError(f"Missing neuronNumber for {area} {pop}")

                csv_path = os.path.join(area_path, fname)
                neuron_num = neuronNumber[area][pop]

                fr = compute_population_firing_rate(
                    csv_path,
                    neuron_num,
                    SIMULATION_TIME_S
                )

                records.append({
                    "version": version,
                    "area": area,
                    "population": pop,
                    "firing_rate": fr
                })

    return pd.DataFrame.from_records(records)


# =========================
# 运行
# =========================

if __name__ == "__main__":
    df = collect_firing_rate_data()

    # print(df.head())
    # print("\nSummary:")
    # print(df.groupby(["version", "area", "population"])["firing_rate"].describe())

    # 保存，方便后续画图 / 统计
    df.to_csv("firing_rate_comparison.csv", index=False)

    # baseline = (
    #     df[df.version == "sGPU"]
    #     .groupby("population")["firing_rate"]
    #     .median()
    # )
    # df["firing_rate_norm"] = df.apply(
    #     lambda r: r.firing_rate / baseline[r.population],
    #     axis=1
    # )
    pop_order = sorted(df["population"].unique())
    plt.figure(figsize=(12, 6))

    sns.violinplot(
        data=df,
        x="population",
        y="firing_rate",
        hue="version",
        order=pop_order,
        split=True,

        density_norm='width',      # 🔑 让翅膀更宽
        cut=0,              # 🔑 不让 KDE 超出数据范围
        bw_adjust=0.6,      # 🔑 降低极端值对 KDE 的影响
        
        inner="box",        # 🔑 用 box 而不是 quartile
        linewidth=1
    )

    # sns.stripplot(
    #     data=df,
    #     x="population",
    #     y="firing_rate",
    #     hue="version",
    #     order=pop_order,
    #     dodge=True,
    #     color="k",
    #     size=3,
    #     alpha=0.4
    # )

    plt.ylabel("Firing rate (Hz)")
    plt.xlabel("Population")
    plt.title("Population firing rate comparison (merged across areas)")
    plt.legend(title="Version")

    plt.tight_layout()
    plt.savefig(
        os.path.join("population_firing_rate_violin.png"),
        dpi=300
    )
    plt.close()