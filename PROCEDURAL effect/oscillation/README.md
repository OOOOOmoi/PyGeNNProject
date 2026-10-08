# Wang-Buzsaki (1996) ING 模型 — PyGeNN 5.2 实现

论文: Wang X-J & Buzsaki G, "Gamma Oscillation by Synaptic Inhibition in a
Hippocampal Interneuronal Network Model", *J. Neurosci.* 16(20):6402-6413.

与 MATLAB 复现版 (`wb1996_*.m`) 严格对齐的 GPU 实现。

## 文件

| 文件 | 说明 |
|------|------|
| `wb1996_genn.py` | 模型 + 仿真 + 分析 + 绘图, 单文件 |
| `run.sh` | 运行入口 (设置 CUDA 环境变量) |
| `output/` | 仿真输出 (PNG 图 + NPZ 数据) |

## 运行

```bash
./run.sh                        # 默认: N=100, Msyn=100 全连接, T=1000 ms
./run.sh --Msyn 60              # 稀疏连接, 论文 Fig.8 部分同步
./run.sh --Msyn 30 --tag desync # 论文 Fig.9 去同步
./run.sh --Msyn 60 --iapp-std 0.05   # Iapp 异质性
./run.sh --Msyn 40 --gpu 1      # 指定 GPU 1
```

参数: `--Msyn --T --dt --iapp --iapp-std --seed --gpu --tag`, 详见
`python wb1996_genn.py --help`。

## 模型要点

- **神经元**: 单室 HH (`m=m_inf` 瞬态), `phi=5`; 无硬重置, 用 spiked 状态位
  实现"上穿 -20 mV 触发一次尖峰"。
- **突触**: GABAA 分级释放 `F(V_pre)=1/(1+exp(-V_pre/2))`,
  `ds/dt = alpha*F*(1-s) - beta*s` (alpha=2.0/ms, beta=0.1/ms)。
- **电导归一化** (论文 p3): 每突触 `w = 0.1/Msyn` mS/cm^2, 总电导恒 0.1。
- **连接**: 每个突触前神经元无放回抽取 Msyn 个突触后目标, 排除自连接
  (与 `wb1996_run_network.m` 一致: 固定出度, 入度 ~ Poisson 涨落)。
  入度涨落提供的结构异质性是 M 越小同步越差的关键;
  若误用固定入度, 各神经元总输入电导完全相同, 任意 M 下都近乎完美同步。
- **矩阵**: DENSE 稠密矩阵 (N=100 仅 10^4 权重), 宿主机侧 numpy 生成。

## GeNN 实现要点 (移植时踩过的坑)

1. `synapse_dynamics_code` 里 `addToPost` 的 inSyn **不会**被自动清零
   (连续动力学 WUM)。必须在 PSM `sim_code` 末尾手动 `inSyn = 0.0`
   (DeltaCurr 模式), 否则电导跨步累积、抑制爆炸。
2. WUM 访问突触前膜电位: `pre_neuron_var_refs` + `pre_var_refs` 绑定;
   PSM 访问突触后 V: `neuron_var_refs` + `var_refs` 绑定。
3. SPARSE + `set_sparse_connections` 有 bug: 该方法存紧凑 ind, 但生成的
   内核按 num_post 填充步长寻址, 行号 ≥ 紧凑长度/步长 后读到未初始化内存
   → cuda error 700。自定义连接片段转译器又不支持运行时长度数组。
   故 N=100 直接用 DENSE 稠密矩阵; 更大 N 需换 FixedNumberPostWithReplacement
   或修复稀疏路径。
4. `.bashrc` 中 CUDA export 行粘连, 非交互 shell 下 CUDA_PATH 未设置,
   `run.sh` 中显式 export。
5. 远程无中文字体, 图内文字一律英文。
6. 同步脉冲状 PSTH 的 FFT 全局峰值常落在 2/3 次谐波 (88/132 Hz),
   分析时必须单独报告 Gamma 带内峰值。

## 输出

- `output/wb1996_genn_<tag>.png`: 2x2 面板 (raster / 群体放电率 /
  PSTH 功率谱+Gamma 带标注 / 示例膜电位)
- `output/wb1996_genn_<tag>.npz`: spike 时间、V 轨迹、PSTH、频谱、
  参数与指标 (mean_rate, peak_freq, gamma_peak_freq, gamma_ratio, kappa)

## 实测结果 (A30, dt=0.02 ms, T=1000 ms, seed=1996)

| Msyn | 状态 | kappa(5ms) | Gamma 带内峰值 | MATLAB 对照 kappa |
|------|------|-----------|---------------|------------------|
| 100 (全连接) | 完美零相位同步 | 0.988 | 44.4 Hz | 1.000 |
| 60 | 部分同步 (Fig.8) | 0.691 | 44.4 Hz | 0.726 |
| 30 | 去同步 (Fig.9) | 0.061 | 43.3 Hz | 0.024 |

平均放电率 ~44 Hz (MATLAB 42.5 Hz), 仿真速度 ~660 ms/s 墙钟。
