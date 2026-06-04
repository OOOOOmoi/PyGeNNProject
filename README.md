# PyGeNNProject — 猕猴视觉皮层大规模脉冲神经网络仿真平台

基于 **PyGeNN**（GPU-enhanced Neural Networks）+ **Numba CUDA** + **多进程/MPI 并行** 的大规模脉冲神经网络（Spiking Neural Network）仿真平台，用于模拟猕猴（Macaque）大脑视觉皮层的神经元群体活动。

---

## 项目概述

本项目构建了从单神经元、单皮层柱到多脑区全视觉皮层的大规模 SNN 模型，涵盖 32 个视觉皮层区域，支持多 GPU 并行加速。模型基于 Schmidt 等人的猕猴视觉皮层微电路框架，并通过 PyGeNN 实现高效的 GPU 仿真。

## 模型目录

| 目录 | 说明 |
|------|------|
| `SingleNeuron/` | 单神经元 LIF 模型测试 |
| `expLIF/` | Exponential LIF 神经元模型定义（带自适应指数） |
| `DualEXP/` | 双指数突触（Dual Exponential Synapse）模型 |
| `IzhkNeuron/` | Izhikevich 神经元模型实现 |
| `EIBalance/` | 兴奋-抑制平衡网络模型 |
| `SingleColumn/` | 单皮层柱模型（Potjans-Diesmann 风格，16 种神经元群体） |
| `DoubleColumn/` | 双皮层柱互连模型 |
| `MultiColumn/` | 多皮层柱扩展模型 |
| `MultiLayer/` | 多层网络模型 |
| `potjansModel/` | Potjans-Diesmann 2014 皮层微电路复现 |
| `SchmidtModel/` | Schmidt 猕猴视觉皮层微电路模型 |
| `CustomModel/` | 自定义全皮层连接模型（支持 32 脑区） |
| `HMAM/` | 层次化多脑区模型（Hierarchical Multi-Area Model） |
| `HMAM_MPI/` | HMAM 的 MPI 分布式多机版本 |
| `MAM_MPI/` | **主干模型**：多脑区 MPI 并行模型（32 脑区，多 GPU 多进程） |
| `ScalingTest/` | 模型规模可扩展性测试 |
| `ProjectOfStimExcNeuronInFirstArea/` | 第一脑区兴奋性神经元刺激实验 |
| `model_info_schmidt_motif_diff_s/` | Schmidt motif 模型的连接数据文件 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 神经元模型 | LIF / Exponential LIF / Izhikevich |
| 突触模型 | ExpCurr（指数衰减电流）/ Dual Exponential |
| GPU 加速 | PyGeNN + Numba CUDA（自定义核函数） |
| 并行框架 | Python `multiprocessing`（多 GPU）+ MPI（多机） |
| 数据处理 | NumPy, SciPy, Pandas |
| 可视化 | Matplotlib, Seaborn |
| 构建系统 | setuptools |

## 主干模型：MAM_MPI

MAM_MPI 是本项目最完整的仿真模型，模拟 **32 个猕猴视觉皮层区域**（V1, V2, V3, V4, MT, MSTd, LIP, FEF, 7a, 46, AITd 等），每个区域包含 5 个皮层层次和 4–5 种神经元亚型。

```bash
# 基本运行
python MAM_MPI/CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0

# 带刺激运行
python MAM_MPI/CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0 --stim --stim-start 300 --stim-end 800
```

详见 `MAM_MPI/readme.md`。

## 依赖

- Python 3.8+
- [PyGeNN](https://github.com/genn-team/genn)（GPU SNN 仿真框架）
- Numba（CUDA JIT 编译）
- NumPy, SciPy, Pandas, Matplotlib
- `nested_dict`, `pynvml`

```bash
pip install pandas matplotlib scipy seaborn numba nested_dict pynvml
```

## 安装

```bash
git clone git@github.com:OOOOOmoi/PyGeNNProject.git
cd PyGeNNProject
pip install -e .
```

## 数据文件

- `custom_Data_Model_3396.json` — 32 脑区完整连接参数（3396 mm² 皮层表面积）
- `default_Data_Model__*.json` — 不同皮层表面积（50/70/100 mm²）的模型参数
- `indegrees_full.json` / `outdegrees_full.json` — 全脑出入度数据
- `viscortex_raw_data.json` — 视觉皮层原始连接数据
- `Fac_result.json` — 促进因子结果

## 参考文献

- Schmidt et al., "A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque visual cortical areas", *PLOS Computational Biology*, 2018.
- Potjans & Diesmann, "The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model", *Cerebral Cortex*, 2014.
- Knight & Nowotny, "GPUs outperform current HPC and neuromorphic solutions in terms of speed and energy when simulating a highly-connected cortical model", *Frontiers in Neuroscience*, 2018.
