# V1 单脑区抑制微环路调参 — 改动与结论总结

> 更新时间：2026-11-11
> 范围：MAM_MPI 32 区猕猴视觉皮层 SNN（Schmidt 2018）中 V1 单脑区的抑制性中间神经元（PV/SST/VIP）微环路调参
> 目标：定位并修复 Allen 小鼠数据驱动的皮层微环路中，抑制性群体（尤其 VIP/SST）放电失衡的问题

---

## 一、核心代码改动

### 1. 突触权重体系对齐原版 Schmidt（`MAM_MPI/getStruct.py`）

`getWeightMap_full_type` 加入生理权重因子，替换原来的简化写法：

```python
PSP_e   = connection_params['PSP_e']    # 兴奋性皮层内突触 PSP (mV)
PSP_ext = connection_params['PSP_ext']  # 外部输入突触权重 (mV)
g       = connection_params['g']        # 抑制相对兴奋强度 (负值, 默认 -16)

# 兴奋 (srcPop[0]=='E'):  PSC_over_PSP * PSP_e
# 抑制 (H/P/S/V):          PSC_over_PSP * g * PSP_e
# 外部输入:                PSC_over_PSP * PSP_ext
```

同时**删除了原代码中硬编码的 S 层自连接放大因子**（S23×10、S4×7、S5×12），使权重体系回归到由 `g=-16` 统一决定的干净形式。

### 2. 外部输入按群体类型缩放（`MAM_MPI/CustomModel_MPI.py`）

新增三个命令行参数，用于细粒度调节外部 Poisson 输入：

| 参数 | 作用 |
|---|---|
| `--ext-scale` | 全局外部输入速率缩放 |
| `--vip-ext-scale` | 仅 VIP（V 型，前缀 'V'）群体的外部输入缩放 |
| `--sst-ext-scale` | 仅 SST（S 型，前缀 'S'）群体的外部输入缩放 |

外部输入速率计算：

```python
type_scale = 1.0
if pop.startswith('V'):  type_scale = vip_ext_scale
elif pop.startswith('S'): type_scale = sst_ext_scale
eff_scale = ext_scale * type_scale
rate = SN[area][pop]["external"]["external"] / NN[area][pop] * eff_scale
```

### 3. 去掉外部输入的 `/3000` 缩放

原代码 `rate = SN/NN/3000*ext_scale` 中 `/3000` 掩盖了爆发放电，已去除。

### 4. 新增 `--data-json` 参数

支持显式指定 Data_Model json 路径（用于加载 balanced 连接数据），model_name 加 `_balancedjson` 标识。

---

## 二、数据侧改动：I→I 连接平衡化

- 脚本：`不需要文件/model-lgq/mam-allen/multiarea_model/data_multiarea/balance_inhibitory.py`（本地）
- 生成：`model_info_schmidt_motif_diff_s/default_Data_Model__balanced_1.0mm2.json`
- 原则：**只放大不缩小**抑制性连接概率，对齐 Potjans E/I 入度比 3.6~5.1，0 概率格子补 0.05，保留原始概率以便切回。

---

## 三、关键实验发现

### 发现 1：各群体外部输入速率严重不均（VIP 偏高）

每神经元外部输入速率 `ext_syn / N`（单位 Hz）：

| 群体类型 | 速率 | 备注 |
|---|---|---|
| E4/E23 | 4072.6 | 兴奋 |
| E5 | 2443.6 | 兴奋 |
| H1/P 系列 | 2036.3 | — |
| **V4/V5/V6** | **6108.9** | **VIP，最高** |
| S 系列 | 610.9 | SST，最低 |

**VIP 外部驱动是 E 层的 1.5~3 倍、SST 的 10 倍**，是 V 层爆发的直接原因。

### 发现 2：VIP 外部刺激存在陡峭相变点（vipext ≈ 0.31）

细粒度扫描（vipext = 0.26~0.55，步长 0.025）：

- **V4 在 vipext 0.30→0.325 之间瞬间跳变 18 倍**（1.52 → 28.27 Hz）
- **S4 同步反向暴跌 5 倍**（63.29 → 12.37 Hz）
- V4 与 S4 呈完美镜像"跷跷板"，无平滑过渡 → 典型的 **E-I 双稳态 / 迟滞（hysteresis）**

### 发现 3：VIP↔SST 竞争性互抑（winner-take-all）

二维联合扫描（vipext × sstext）揭示：VIP 和 SST 之间存在**双向抑制**（不只 VIP→SST，还有 SST→VIP）：

- 压 VIP → SST 去抑制释放 → **S4 爆发**（60~73 Hz）
- 压 SST → VIP 去抑制释放 → **V4 高位**（30~48 Hz）
- 两个抑制性群体**无法同时处于中等活动水平**，总有一个被推高、一个被压低

### 结论

这个网络的核心问题是 **VIP 与 SST 之间的互抑太强、太对称**，形成 winner-take-all 竞争。单边调外部输入只在两个"赢家"间切换，无法实现平衡共存。

**可能的修复方向**：
1. 同时降低 VIP + SST 外部输入（双降），让二者回到网络内部连接决定的平衡态
2. 调整 VIP↔SST 之间的突触权重（互抑强度），破坏对称竞争

---

## 四、并行扫描基础设施

- 单区 V1 仿真：213481 神经元、4.08 亿突触，1 个 worker 进程
- 编译目录 `worker0_on_device{gpu}` 固定 → 同卡不能并发多参数组合
- 分批并行：每批 N 个组合分配到不同 GPU，`wait` 后下一批
- 扫描脚本（`MAM_MPI/` 下）：
  - `runner_v1_ext_scan.sh` — ext-scale 扫描
  - `runner_v1_inh_scan.sh` — inh-scale 扫描
  - `runner_v1_vipext_scan.sh` / `runner_v1_vipext_fine.sh` — vip-ext-scale 扫描
  - `runner_v1_2d_scan.sh` — vipext × sstext 二维扫描
- 分析脚本：`analyze_*.py`（群体放电率统计）、`plot_raster_scan.py`（raster plot 生成，输出 `output/raster_scan/`）

### 命名约定注意

- `model_name` 由 `getModelName` 用 float 直接格式化，`0.010` → `ext0.01`、`0.30` → `vipext0.3`
- 参数值为默认值（如 `ext_scale=1.0`、`sst_ext_scale=1.0`）时不加后缀
- 分析脚本解析目录名时需注意去尾零

---

## 五、相变来源定位实验（2026-10-28 ~ 2026-11-11）

> 目标：回答"vipext 相变点（发现 2）到底由什么机制驱动"。此前假设是 VIP↔SST 对称互抑构成的 winner-take-all 双稳态，后续实验逐步证伪并改写结论。

### 实验 A：gL 放大（VIP f-I 增益）→ 平移而非软化

- 新增 `--vip-gl-scale` 参数，仅缩放 VIP（V 型）群体的漏电导 gL，降低膜电阻与 f-I 增益。
- 结果：gL×2 只是**平移**相变边界，不软化。0.01 步长细扫精确定量：gL=2.0 使 V4/S4 相变边界整体下移 **-0.0525±0.0144**（vipext 单位），V4/S4 完全同步。
- **结论**：单神经元 f-I 增益不是相变来源（若是"放大器/开关"，调增益应软化斜率）。相变来自网络层面。

### 实验 B：第四层 SST↔VIP 互抑权重二维联合扫描（证伪对称双稳态）

- 新增 `--sst2vip-scale`（S4→V4）与 `--vip2sst-scale`（V4→S4），仅缩放第四层互抑权重，二维扫描 21×21=441 组合（0~1 步长 0.05）。
- 四个对照点结果：

| 群体 | 全关断(0,0) | 只关S2V(0,1) | 只关V2S(1,0) | 基线(1,1) |
|---|---|---|---|---|
| S4 (SST) | **12.01** | 0.24 | **12.44** | 0.29 Hz |
| V4 (VIP) | 97.01 | 93.73 | 79.91 | 93.38 Hz |
| E4 | 5.18 | 4.38 | 6.97 | 4.43 Hz |
| P4 | 14.26 | 22.12 | 14.50 | 22.12 Hz |

- **三个关键发现**：
  1. 只有 S4 有相变，V4/E4/P4 几乎不动（V4 全程 80~97 Hz）。
  2. S4 的相变**完全由 VIP→SST 单向通路决定，且是连续渐变不是陡跳**（12→9.3→6.9→4.9→2.8→0.9→0.24 Hz 平滑下降）。
  3. SST→VIP 通路在当前网络里几乎没有功能作用（调 s2v 从 0→1，S4 不变、V4 只微降 13%）。
- **结论（推翻旧假设）**：不是对称 winner-take-all 双稳态，而是 **VIP→SST 单向单调抑制**。之前 vipext 相变的"陡峭"来自 VIP 的兴奋输入侧（外部 + E→VIP），不是 SST↔VIP 之间的对称反馈。

### 实验 C：全局外部输入 ext-scale 精细扫描（0~0.1 步长 0.001）

- 纯扫 `--ext-scale`（全局外部输入乘子，所有群体同乘），其它参数默认 1.0，101 组合。
- 五种类型分化（ext=0.1 时放电率）：**V(VIP) ~260 Hz > P(PV) ~60-158 > H ~48 > E ~13-50 > S(SST) ~0.4-8**。
- **SST 是唯一"先升后降"的类型**：S4 在 ext≈0.005 处峰值 2.6 Hz 后被压到 0.43 Hz；S23/S5 在 ext=0 时反而最高（9~10 Hz），输入一进来就暴跌。
- **衔接**：VIP 是外部输入的高增益放大器 → 经 VIP→SST 单向压制 → SST 随输入单调/非单调下降。进一步确认相变源头在 VIP 兴奋输入侧。

### 修正后的核心结论（取代旧"发现 3"）

网络的关键问题**不是** VIP↔SST 互抑太强太对称的 winner-take-all 竞争，而是：

1. **VIP 外部驱动过高**（6108.9 Hz，是 E 层 1.5~3 倍、SST 的 10 倍）+ VIP 本身高 f-I 增益 → VIP 易爆发。
2. **VIP→SST 单向强抑制** → VIP 一旦被推高，SST 就被单调压制，二者无法共存。
3. 相变的"陡峭跳变"来自 VIP 兴奋输入侧的放大（外部 + E→VIP 通路），而非 SST↔VIP 的对称反馈双稳态。

**下一步方向**（未做）：
- 扫 E→VIP 兴奋权重，看是否与 ext-scale 一样能复现 V4 陡升 + S4 跌落。
- 检查 VIP 高放电率(260 Hz)是否生理合理，是否需引入 VIP 的强自抑制或其它去放大机制。

---

## 六、脚本目录整理（2026-11-11）

- 扫描/分析脚本已从 `MAM_MPI/` 根目录归纳到 `MAM_MPI/scans/` 子目录：
  - `runner_v1_*.sh`（19 个，含 runner_verify_*.sh、test_compile_reuse.sh）
  - `analyze_*.py`（18 个）
  - `plot_*.py`（8 个）
- 根目录仅保留核心程序与配置：`CustomModel_MPI.py`、`config.py`、`getStruct.py`、`connectom.py`、`runner.sh`、`runner_multi.sh` 等。
- 移动后路径修复：
  - runner 脚本 `cd "$(dirname "$0")"` → `cd "$(dirname "$0")/.."`（回到 MAM_MPI 根目录调用 CustomModel_MPI.py）。
  - `analyze_connect.py`、`analyze_rate.py` 的 `../model_info_...` → `../../model_info_...`。
- 其余 analyze/plot 脚本用绝对路径 `/home/yangjinhao/PyGeNNProject/...`，移动后无需改动。
