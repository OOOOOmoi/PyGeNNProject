%% ================================================================
%% CustomModel_MPI.m  —  MATLAB 版本（单 Worker 建模过程）
%% 从 CustomModel_MPI.py 转换而来，保留建模 + 单 Worker 仿真逻辑。
%% 跳过了 MPI 多进程通信部分（split_indices, Queue, 主进程循环等）。
%% 不可直接运行，用于理解和分析模型构建与算法。
%% ================================================================

clear; clc;

%% ===================== 全局参数 =====================
DT_MS             = 0.1;       % 仿真时间步长 (ms)
NUM_THREADS_PER_SPIKE = 1;     % Procedural 矩阵每 spike 线程数
MAX_SHARED_BINS   = 1024;      % GPU shared memory 最大 bins 数
buffer_size       = 1;         % GeNN 录制缓冲区大小

%% ===================== 1. 数据加载（Python: prepare()）=====================
% DataPath = fullfile(parent_dir, 'model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json');
% ParamOfAll = jsondecode(fileread(DataPath));
%   SynapsesNumber   — 突触数量 [tar_area][tar_pop][src_area][src_pop]
%   NeuronNumber     — 神经元数量 [area][pop]
%   Dist             — 距离矩阵
%   area_list        — 脑区列表 (32个)
%   pop_list         — 群体类型列表 (17种)
%   SynapsesWeightMean, SynapsesWeightSd — 权重均值/标准差
%   delayMap         — 延迟映射 [tar_area][tar_pop][src_area][src_pop]{'ave','sd','max'}
%   Ind              — V1 区归一化因子

area_list = {'V1','V2','VP','V3','V3A','MT','V4t','V4','VOT','MSTd', ...
             'PIP','PO','DP','MIP','MDP','VIP','LIP','PITv','PITd', ...
             'MSTl','CITv','CITd','FEF','TF','AITv','FST','7a', ...
             'STPp','STPa','46','AITd','TH'};

pop_list = {'H1','E23','S23','P23','V23','E4','S4','P4','V4', ...
            'E5','S5','P5','V5','E6','S6','P6','V6'};

num_areas = 32;       % AreaNum
num_workers = 32;     % Worker 数量（每个 worker 1 个脑区）


%% ===================== 2. 参数查表（Python: config.py）=====================
% 以下参数结构同 CustomModel.py，此处仅列关键字段，详见 config.py

Cm_keys   = {'H1','E23','P23','S23','V23','E4','P4','S4','V4','E5','P5','S5','V5','E6','P6','S6','V6'};
Cm_vals   = [37.11,123.41,70.95,82.34,41.23,80.16,81.21,132.86,40.3,149.43,70.9,52.32,59.29,99.96,49.65,96.09,65.87];
Cm = containers.Map(Cm_keys, Cm_vals);

gL_vals = [4.07,2.47,9.49,3.17,6.4,5.16,9.19,7.96,1.87,16.66,5.21,3.43,6.52,5.88,6.86,2.99,6.09];
gL = containers.Map(Cm_keys, gL_vals);

tref_vals = [3.5,3.0,1.26,1.85,2.75,4.4,1.5,2.2,2.4,4.25,1.85,1.9,2.55,3.3,1.65,2.1,2.85];
tref = containers.Map(Cm_keys, tref_vals);

Vrest_vals = [-65.5,-80.97,-82.35,-69.16,-67.94,-72.53,-70.45,-74.2,-63.14,-68.28,-77.5,-70.01,-72.0,-77.5,-76.42,-62.99,-78.85];
Vrest = containers.Map(Cm_keys, Vrest_vals);

Vth_vals = [-40.20,-40.53,-56.32,-39.95,-41.34,-47.63,-44.23,-44.07,-40.89,-40.55,-51.2,-47.38,-51.2,-42.31,-49.06,-37.19,-44.81];
Vth = containers.Map(Cm_keys, Vth_vals);

% 通用默认值
E_L   = -70.0;         % mV
tau_syn = 0.5;          % ms（突触后时间常数）
rate_ext = 10.0;        % Hz（外部输入频率）
scale_ = 1.0;           % 权重缩放因子


%% ===================== 3. spike 缓冲区构建（Python: build_spike_buffer()）=====================
% 这是 MPI 版本的核心数据结构：为每个 (target, source) 突触群对建立一个
% 环形缓冲区，存储延迟到达的 spike count。同时预计算连接概率、权重等数组，
% 供 GPU 核函数快速查表使用。

% 输入（实际从 JSON 加载）:
%   NN        — NeuronNumber [area][pop], 标量
%   SN        — SynapsesNumber [tar_area][tar_pop][src_area][src_pop], 标量
%   delay_cc  — delayMap, 包含 'ave' 字段
%   weight    — SynapsesWeightMean [tar_area][tar_pop][src_area][src_pop]
%   tar_area_list — 当前 worker 负责的脑区列表
%   all_area  — 全部脑区列表（前 area_num 个被当作源脑区）

fprintf('\n=== spike 缓冲区构建（build_spike_buffer）===\n');
fprintf('为每个跨区 (target, source) 突触群对创建环形 buffer。\n');
fprintf('同时构建以下辅助数组（供 GPU kernel 使用）：\n');
fprintf('  weight_array      — 每个源突触群的权重均值 (n_src,)\n');
fprintf('  prob_array        — 每个源突触群的连接概率 (n_src,)\n');
fprintf('  src_pop_num_array — 每个目标群接收的源群数量 (n_groups,)\n');
fprintf('  tar_neu_num_array — 每个目标群的神经元数量 (n_groups,)\n');
fprintf('  R_array           — 每个目标神经元的膜阻 Rm (total_tar,)\n');
fprintf('\n算法伪代码:\n');
fprintf('  for tar_area in tar_area_list:\n');
fprintf('    for tar_pop in pop_list:\n');
fprintf('      Rm = 1 / gL[tar_pop] * 1000;  %% 膜阻 (MOhm)\n');
fprintf('      for src_area in all_area[0:area_num]:\n');
fprintf('        for src_pop in pop_list:\n');
fprintf('          跳过 src_area==tar_area 或零连接;\n');
fprintf('          prob = conn_num / N_src / N_tar;\n');
fprintf('          delay_step = ceil(delay_ms / dt);\n');
fprintf('          buffer[(tar,src)] = zeros(delay_step);\n');
fprintf('          收集 weight, prob, src_pop_num, tar_neu_num, Rm;\n');
fprintf('\n');


%% ===================== 4. GPU 核函数：fast_update_inSyn（Python: numba.cuda.jit）=====================
% 这是 MPI 版本最关键的算法。它用 GPU 并行化突触输入的随机生成：
% 每个 block 处理一个源突触群 (sc)，生成 sc * group_neu_count 次伯努利试验，
% 命中则随机选取目标神经元并累加随机权重。

fprintf('\n=== GPU Kernel: fast_update_inSyn_gpu ===\n');
fprintf('每个 CUDA block 处理一个源突触群 (source connection group)。\n');
fprintf('\n输入:\n');
fprintf('  spike_array     — 各源群的本步 spike count [n_src]\n');
fprintf('  cum_src         — 源群数量累积和（二分查找定位 group）[n_groups]\n');
fprintf('  tar_neu_num_array — 各目标群神经元数 [n_groups]\n');
fprintf('  prob_array      — 连接概率 [n_src]\n');
fprintf('  weight_array    — 权重均值 [n_src]\n');
fprintf('  inSyn_buffer    — 突触输入缓冲区（被原子更新）[sum_tar]\n');
fprintf('  rng_states      — XOROSHIRO128+ 随机数状态 [blocks*threads]\n');
fprintf('\n算法流程（每 block）:\n');
fprintf('  1. 获取本 block 对应的 spike count sc;\n');
fprintf('  2. 二分查找确定所属 target group;\n');
fprintf('  3. n_trials = sc * group_neu_count;\n');
fprintf('  4. 小规模 (<2048): 每线程 Bernoulli 累加 + block 内归约;\n');
fprintf('  5. 大规模 (>=2048): tid==0 用 Poisson 近似 (Knuth''s algorithm);\n');
fprintf('  6. 用 total_hits 次随机抽样选取目标神经元;\n');
fprintf('  7. 权重 = weight + 0.1*weight*Normal(0,1), clamp >= 0;\n');
fprintf('  8. 原子加 (atomic.add) 写入 inSyn_buffer;\n');
fprintf('\n');


%% ===================== 5. GPU 核函数：decay（Python: decay_gpu）=====================
fprintf('\n=== GPU Kernel: decay_gpu ===\n');
fprintf('对 inSyn_buffer 的每个元素做指数衰减:\n');
fprintf('  inSyn[i] = inSyn[i] * exp(-dt / 0.5)\n');
fprintf('（MPI 版本中衰减改为 CPU 端执行，GPU 只做清零以准备下一步）\n');
fprintf('\n');


%% ===================== 6. 模型初始化（Python: Part() 中的 GeNNModel）=====================
fprintf('\n=== Worker 模型初始化 ===\n');

% Python:
%   model = GeNNModel("float", "GenCODE/worker{id}_on_device{gpu_id}",
%                      device_select_method=DeviceSelect.MANUAL, manual_device_id=gpu_id)
%
%   model.dt = 0.1;
%   model.fuse_postsynaptic_models = true;            % 融合突触后模型
%   model.default_narrow_sparse_ind_enabled = true;   % 窄稀疏索引
%   model.timing_enabled = true;                      % 计时
%   model.default_var_location = VarLocation.HOST_DEVICE;
%   model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE;

model_cfg = struct();
model_cfg.precision       = 'float';
model_cfg.dt              = DT_MS;       % 0.1 ms
model_cfg.fuse_postsynaptic = true;
model_cfg.narrow_sparse   = true;
model_cfg.timing_enabled  = true;
model_cfg.var_location    = 'HOST_DEVICE';
model_cfg.sparse_loc      = 'HOST_DEVICE';

fprintf('  精度:              %s\n', model_cfg.precision);
fprintf('  时间步长:           %.1f ms\n', model_cfg.dt);
fprintf('  融合突触后模型:     %d\n', model_cfg.fuse_postsynaptic);
fprintf('  窄稀疏索引:         %d\n', model_cfg.narrow_sparse);


%% ===================== 7. 突触后模型 & 电流源 ======================
fprintf('\n=== 突触后模型 & 电流源 ===\n');

% ExpCurr 突触后模型 (兴奋/抑制各一个)
%   exc_exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5});
%   inh_exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5});
fprintf('  突触后模型: ExpCurr (tau=0.5 ms)\n');
fprintf('  区分兴奋/抑制，但参数相同\n');

% TriggerPulse 电流源（刺激用）
%   trigger_pulse_model = pygenn.create_current_source_model(
%       "trigger_pulse",
%       params=["start_time","end_time","magnitude"],
%       injection_code = "if (t >= start_time && t < end_time) { injectCurrent(magnitude); }"
%   );
fprintf('  电流源1: TriggerPulse — 时间窗口内注入恒定电流\n');

% PoissonExp 电流源（背景噪声）
%   poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": rate};
%   model.add_current_source(popName + "_poisson", "PoissonExp", neuron_pop, poisson_params, poisson_init);
fprintf('  电流源2: PoissonExp — 泊松背景输入\n');
fprintf('    权重: ext_weight = weight[area][pop][''external''][''external'']\n');
fprintf('    频率: rate = SN[area][pop][''external''][''external''] / NN[area][pop] / 3000\n');
fprintf('\n');


%% ===================== 8. 神经元群体创建 ======================
fprintf('\n=== 神经元群体参数 ===\n');
fprintf('模型类型: LIF (Leaky Integrate-and-Fire)\n');
fprintf('%-15s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Pop', 'C(nF)', 'TauM(ms)', 'Vrest(mV)', 'Vreset(mV)', 'Vth(mV)');

for p = 1:length(pop_list)
    pop = pop_list{p};
    C_val  = Cm(pop) / 1000.0;           % nF
    TauM   = Cm(pop) / gL(pop);           % ms
    V_rest = Vrest(pop);
    V_res  = Vrest(pop) - 10.0;
    V_th   = Vth(pop);
    T_r    = tref(pop);
    
    fprintf('%-15s %-10.4f %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
        pop, C_val, TauM, V_rest, V_res, V_th); %#ok<*NASGU>
end

fprintf('\n初始条件:\n');
fprintf('  V:           Normal(mean=-150.0, sd=50.0) [mV]\n');
fprintf('  RefracTime:  = TauRefrac [ms]\n');
fprintf('  Ioffset:     0 [nA]\n');
fprintf('  Spike 录制:  启用\n');


%% ===================== 9. 突触群体创建 ======================
fprintf('\n=== 突触连接创建 ===\n');
fprintf('权重更新模型:  StaticPulseDendriticDelay\n');
fprintf('权重分布:      NormalClipped\n');
fprintf('  — 兴奋性 (popSrc 以 ''E'' 开头): [0, +inf)\n');
fprintf('  — 抑制性:                         (-inf, 0]\n');
fprintf('延迟分布:      NormalClippedDelay\n');
fprintf('  — mean, sd, min=0, max=max_d (从 delayMap 取)\n');
fprintf('稀疏连接:      FixedNumberTotalWithReplacement\n');
fprintf('  — 每个突触群 synNum 个连接，有放回抽样\n');
fprintf('矩阵类型:      PROCEDURAL (默认, num_threads_per_spike=1)\n');
fprintf('\n');

% 示例：V1_E4 -> V1_E23 连接参数
fprintf('=== 示例: V1_E4 -> V1_E23 ===\n');
wAve_ex  = 0.15 / 1000.0;    % uS
wSd_ex   = 0.015 / 1000.0;
d_mean   = 1.5;  % ms
d_sd     = 0.5;
d_max    = 3.0;
synNum_ex = 1000000;

fprintf('  权重:    %.6f +/- %.6f uS (NormalClipped [0,+inf))\n', wAve_ex, wSd_ex);
fprintf('  延迟:    %.2f +/- %.2f ms, max=%.2f, steps=%d\n', ...
    d_mean, d_sd, d_max, round(d_max/DT_MS));
fprintf('  突触数:  %d (FixedNumberTotalWithReplacement)\n', synNum_ex);
fprintf('  最大树突延迟步数: %d\n', round(d_max/DT_MS));
fprintf('\n');


%% ===================== 10. 仿真主循环（Python: while model.t < duration）=====================
fprintf('\n=== 单 Worker 仿真循环 ===\n');
fprintf('每步 (dt=0.1ms) 执行以下操作:\n');
fprintf('\n');
fprintf('--- 有跨区连接时 (spike_count_buffer 非空) ---\n');
fprintf('  1. 读取 spike buffer 当前槽的 spike count，立即清零;\n');
fprintf('  2. fast_update_inSyn_gpu  %% GPU 随机突触输入更新\n');
fprintf('     - d_inSyn 拷贝到 host，乘以 exp(-dt/1) 累加到 inSyn_buffer\n');
fprintf('  3. model.step_time()       %% GeNN 推进神经元 + 内部突触\n');
fprintf('  4. 更新膜电位:\n');
fprintf('     - pull V, RefracTime from device\n');
fprintf('     - dV = inSyn * Rm * dt / TauM\n');
fprintf('     - dV[RefracTime > 0] = 0  %% 不应期跳过\n');
fprintf('     - V += dV\n');
fprintf('     - push V to device\n');
fprintf('  5. 衰减 inSyn_buffer *= exp(-dt/0.5); d_inSyn 清零\n');
fprintf('  6. pull_recording_buffers + record_spike\n');
fprintf('  7. 发送精简 spike_count 给主进程（MPI 通信）\n');
fprintf('  8. 从主进程接收全局聚合 spike_count，写回 buffer 延迟槽\n');
fprintf('\n');
fprintf('--- 无跨区连接时 (spike_count_buffer 为空) ---\n');
fprintf('  仅执行 model.step_time() + spike 记录\n');
fprintf('\n');
fprintf('计时日志 (每步记录, ms):\n');
fprintf('  step_time | update_V | update_inSyn | update_decay | update_spike | total\n');
fprintf('\n');


%% ===================== 11. 日志系统 ======================
fprintf('\n=== 日志系统 ===\n');
fprintf('Worker 日志 (worker_{id}.log):\n');
fprintf('  首行: total_neurons,neuron_group,total_synapses,syn_group,build_time,load_time\n');
fprintf('  每步: timestep,step_time,update_V,update_inSyn,update_decay,update_spike,total\n');
fprintf('  末行: total_simulation_time,{time} ms\n');
fprintf('\n');
fprintf('Master 日志 (master.log):\n');
fprintf('  每步每 worker: step,worker_id,latency_ms,speed_MBps,data_size_KB\n');
fprintf('\n');


%% ===================== 12. 模型构建 & 加载 & 运行 ======================
fprintf('\n=== 构建 / 加载 / 运行时序 ===\n');
fprintf('  1. model.build()  — GeNN 代码生成 + CUDA 编译\n');
fprintf('  2. model.load(num_recording_timesteps=buffer_size)\n');
fprintf('     — GPU 显存分配 + 状态初始化\n');
fprintf('  3. build_spike_buffer() — 创建跨区环形 buffer + 辅助数组\n');
fprintf('  4. numba.cuda 准备 — 转移数组到 GPU, 创建 RNG 状态\n');
fprintf('  5. 仿真循环 — while model.t < duration\n');
fprintf('\n');


%% ===================== 13. 主进程逻辑（Python: __main__）=====================
fprintf('\n=== 主进程逻辑（仅描述，MATLAB 不复刻）===\n');
fprintf('  - num_workers = 32, 每个 worker 负责 1 个脑区\n');
fprintf('  - split_indices(32,32) → 每个 worker 1 个 area\n');
fprintf('  - GPU 分配: gpu_id = args.gpu_ids[worker_id %% len(gpu_ids)]\n');
fprintf('  - 主循环每步:\n');
fprintf('      a. 等待所有 worker 提交 spike_count\n');
fprintf('      b. 聚合全局 spike_count\n');
fprintf('      c. 广播给所有 worker (type="continue")\n');
fprintf('      d. 记录 IPC 性能到 master.log\n');
fprintf('  - 仿真完成后发送 type="stop"\n');
fprintf('  - 收集 final_spike_data 并调用 visualize()\n');
fprintf('\n');


%% ===================== 14. 模型结构汇总 ======================
fprintf('%s\n', repmat('=', 1, 55));
fprintf('         CustomModel_MPI  模 型 结 构 汇 总\n');
fprintf('%s\n', repmat('=', 1, 55));
fprintf('  脑区数:              %d (每个 worker 负责 1 个)\n', num_areas);
fprintf('  Worker 数:           %d\n', num_workers);
fprintf('  群体类型:            %d 种\n', length(pop_list));
fprintf('  神经元模型:          LIF\n');
fprintf('  突触后模型:          ExpCurr (tau=0.5 ms)\n');
fprintf('  突触权重更新:        StaticPulseDendriticDelay\n');
fprintf('  权重分布:            NormalClipped\n');
fprintf('  延迟分布:            NormalClippedDelay\n');
fprintf('  连接方式:            FixedNumberTotalWithReplacement\n');
fprintf('  矩阵类型:            PROCEDURAL\n');
fprintf('  电流源:              TriggerPulse / PoissonExp\n');
fprintf('  仿真精度:            float\n');
fprintf('  时间步长:            %.1f ms\n', DT_MS);
fprintf('\n');
fprintf('  核心算法:            GPU 加速随机突触输入\n');
fprintf('    - fast_update_inSyn_gpu  (Numba CUDA kernel)\n');
fprintf('    - Bernoulli/Poisson 近似 + 随机权重 + atomicAdd\n');
fprintf('    - XOROSHIRO128+ 随机数生成器\n');
fprintf('\n');
fprintf('  通信模式:            Master-Worker (Python multiprocessing)\n');
fprintf('    - 每步: worker→master 发送 spike_count\n');
fprintf('    - 每步: master→worker 广播聚合 spike_count\n');
fprintf('    - workers 之间通过 master 交换延迟 spike 信息\n');
fprintf('%s\n', repmat('=', 1, 55));

fprintf('\n转换完成。\n');
fprintf('本脚本涵盖了 CustomModel_MPI.py 中单个 Worker 的完整建模流程和\n');
fprintf('GPU 加速算法描述，跳过了 MPI 多进程通信的实现细节。\n');
