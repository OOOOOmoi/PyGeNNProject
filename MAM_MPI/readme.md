# MAM_MPI — 猕猴视觉皮层大规模脉冲神经网络模拟

基于 **PyGeNN + Numba CUDA + 多进程并行** 的大规模脉冲神经网络（Spiking Neural Network）仿真平台，模拟猕猴（Macaque）大脑视觉皮层的神经元群体活动。

---

## 技术架构

| 层级 | 技术栈 |
|------|--------|
| 神经元模型 | LIF (Leaky Integrate-and-Fire) |
| 突触模型 | ExpCurr（指数衰减电流突触） |
| GPU 加速 | NVIDIA CUDA（Numba 自定义核函数） |
| 并行框架 | Python `multiprocessing`（多 GPU 多进程） |
| 网络建模 | [PyGeNN](https://github.com/genn-team/genn) |
| 可视化 | matplotlib（raster、hist、PSD、放电率曲线） |

## 模型规模

- **32 个视觉皮层区域**：V1, V2, V3, V4, MT, MSTd, LIP, FEF, 7a, 46, AITd 等
- **每个区域含 5 个皮层层次**（Layer 1, 2/3, 4, 5, 6）
- **每层含 4–5 种神经元类型**：E（兴奋性）、S、P、V、H（抑制性），共约 17 种亚型
- 每种神经元类型有独立的生物物理参数（膜电容、漏电导、不应期、静息电位、阈值等）

## 依赖

- Python 3.8+
- [PyGeNN](https://github.com/genn-team/genn)
- Numba（CUDA JIT 编译）
- NumPy, SciPy, Pandas, Matplotlib
- nested_dict

## 快速开始

```bash
# 基本运行（默认参数）
python CustomModel_MPI.py --duration 1000

# 完整运行（32脑区 + 缩放因子 + 突触电流记录）
python CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0 --inSyn

# 带刺激的运行
python CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0 --stim --stim-start 300 --stim-end 800

# 使用不同皮层表面积的模型
python CustomModel_MPI.py --duration 1000 --AreaNum 32 --surface 3396
```

或使用脚本运行：

```bash
bash runner.sh
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--duration` | float | 500.0 | 仿真时长（ms） |
| `--AreaNum` | int | — | 模拟的脑区数量（1–32） |
| `--scale` | float | 1.0 | 跨区域连接权重缩放因子 |
| `--surface` | float | — | 皮层表面积（mm²），影响加载的模型数据文件 |
| `--stim` | flag | False | 是否施加外部刺激 |
| `--stim-start` | float | 300 | 刺激开始时间（ms） |
| `--stim-end` | float | 800 | 刺激结束时间（ms） |
| `--inSyn` | flag | False | 是否记录突触电流和膜电位 |
| `--buffer` | flag | False | 是否使用缓冲区存储 spike |
| `--buffer-size` | int | 100 | 记录缓冲区大小 |

## 项目结构

```
MAM_MPI/
├── CustomModel_MPI.py      # 主程序：模型构建、仿真循环、多进程协调、CUDA 核函数
├── config.py               # 模型参数配置（神经元、突触、连接、刺激）
├── getStruct.py            # 模型结构构建（权重矩阵、延迟矩阵、网络拓扑）
├── visual.py               # 可视化主体：raster 图、放电率、PSD 功率谱
├── record.py               # 数据记录：spike、突触电流、膜电位
├── psd.py                  # 功率谱密度分析
├── rate_curve.py           # 放电率曲线分析
├── connectom.py            # 连接组矩阵绘制
├── inSyn.py                # 突触电流记录函数
├── update_time.py          # 通信耗时可视化
├── read_test.py            # 测试/调试脚本
├── runner.sh               # 运行脚本
├── runner_multi.sh         # 多任务运行脚本
├── GenCODE/                # PyGeNN 生成的 CUDA 代码
├── log/                    # 运行时间详细记录
└── output/                 # 可视化输出（raster、hist、psd、rate）
    └── spike/              # Spike 数据存储（CSV）
```

## 关键技术亮点

### 1. 自定义 Numba CUDA 核函数

`fast_update_inSyn_gpu`：在 GPU 上使用 XOROSHIRO128+ 随机数生成器高效计算泊松噪声突触输入：
- 小规模试验（n_trials < 2048）：精确二项分布采样 + 共享内存归约
- 大规模试验：泊松近似加速

### 2. Spike Buffer 环形缓冲区

处理跨区域轴突传导延迟——非本区域连接通过 spike buffer 在延迟步长后到达目标神经元。

### 3. 多 GPU 多进程并行

- 使用 4 块 GPU，每块 GPU 可运行多个进程
- 主进程通过 `multiprocessing.Queue` 收集各子进程 spike count，聚合后广播
- 仿真结束后统一收集完整 spike 数据进行可视化

### 4. 模型数据文件

模型连接结构（突触数量、神经元数量、距离矩阵）从 JSON 参数文件加载（`model_info_schmidt_motif_diff_s/`），支持不同表面积（如 3396 mm²）的模型变体。

## 数据目录

- **生成代码**：`GenCODE/` — PyGeNN 自动生成的 GPU 代码
- **运行日志**：`log/` — 各 worker 的时间统计（步进、通信、inSyn 更新等）
- **可视化结果**：`output/` — raster、hist、PSD、放电率曲线等

## 参考文献

基于 Schmidt 等人的猕猴视觉皮层微电路模型（microcircuit motif model），在多个皮层区域上扩展为大规模全脑仿真。
