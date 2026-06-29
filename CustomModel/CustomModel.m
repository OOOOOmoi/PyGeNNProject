%% ================================================================
%% CustomModel.m  —  MATLAB 版本（建模过程）
%% 从 CustomModel.py 转换而来，保留完整的建模逻辑。
%% 注意：本脚本不可直接运行，因为需要 GeNN 框架（C++/Python）。
%%       它用于记录模型结构和参数，便于在 MATLAB 环境中理解
%%       和分析模型构建过程。
%% ================================================================

clear; clc;

%% ===================== 全局时间参数 =====================
DT_MS = 0.1;          % 仿真时间步长 (ms)
NUM_THREADS_PER_SPIKE = 1;

%% ===================== 1. 加载数据（Python: prepare()）=====================
% DataPath = fullfile(parent_dir, 'custom_Data_Model_3396.json');
% ParamOfAll = jsondecode(fileread(DataPath));
%
% 从 JSON 中提取的关键数据结构：
%   SynapsesNumber  — 突触数量表  [tar_area][tar_pop][src_area][src_pop]
%   NeuronNumber    — 神经元数量表 [area][pop]
%   Dist            — 距离矩阵
%   area_list       — 脑区列表 (32个视觉皮层相关脑区)
%   pop_list        — 神经元群体类型列表
%
% 以下使用示例数据表示结构（实际运行时从 JSON 加载）

% --- 脑区列表（32个视觉皮层脑区）---
area_list = {'V1','V2','VP','V3','V3A','MT','V4t','V4','VOT','MSTd', ...
             'PIP','PO','DP','MIP','MDP','VIP','LIP','PITv','PITd', ...
             'MSTl','CITv','CITd','FEF','TF','AITv','FST','7a', ...
             'STPp','STPa','46','AITd','TH'};

% --- 神经元群体类型 ---
pop_list = {'H1','E23','S23','P23','V23','E4','S4','P4','V4', ...
            'E5','S5','P5','V5','E6','S6','P6','V6'};

num_areas = length(area_list);
num_pops  = length(pop_list);

fprintf('Areas: %d, Populations: %d\n', num_areas, num_pops);


%% ===================== 2. 神经元参数（Python: config.py -> collection_params）=====================

% --- LIF 单神经元参数（per-population 查表）---
% Cm [pF], gL [nS], tref [ms], Vrest [mV], Vth [mV]
% 这些参数按 pop 类型组织为 containers.Map（MATLAB 的 dictionary）

Cm_keys   = {'H1','E23','P23','S23','V23','E4','P4','S4','V4','E5','P5','S5','V5','E6','P6','S6','V6'};
Cm_vals   = [37.11,123.41,70.95,82.34,41.23,80.16,81.21,132.86,40.3,149.43,70.9,52.32,59.29,99.96,49.65,96.09,65.87];
Cm = containers.Map(Cm_keys, Cm_vals);

gL_keys = Cm_keys;
gL_vals = [4.07,2.47,9.49,3.17,6.4,5.16,9.19,7.96,1.87,16.66,5.21,3.43,6.52,5.88,6.86,2.99,6.09];
gL = containers.Map(gL_keys, gL_vals);

tref_keys = Cm_keys;
tref_vals = [3.5,3.0,1.26,1.85,2.75,4.4,1.5,2.2,2.4,4.25,1.85,1.9,2.55,3.3,1.65,2.1,2.85];
tref = containers.Map(tref_keys, tref_vals);

Vrest_keys = Cm_keys;
Vrest_vals = [-65.5,-80.97,-82.35,-69.16,-67.94,-72.53,-70.45,-74.2,-63.14,-68.28,-77.5,-70.01,-72.0,-77.5,-76.42,-62.99,-78.85];
Vrest = containers.Map(Vrest_keys, Vrest_vals);

Vth_keys = Cm_keys;
Vth_vals = [-40.20,-40.53,-56.32,-39.95,-41.34,-47.63,-44.23,-44.07,-40.89,-40.55,-51.2,-47.38,-51.2,-42.31,-49.06,-37.19,-44.81];
Vth = containers.Map(Vth_keys, Vth_vals);

% 外部输入电流 (DC offset) [pA]
input_keys = Cm_keys;
input_vals = [420,420,420,420,420,420,420,420,420,420,420,420,420,420,420,420,420];
input_current = containers.Map(input_keys, input_vals);

% 通用 LIF 参数（单值）
E_L_default  = -70.0;   % 静息电位 [mV]
V_reset_base = -60.0;   % 重置电位基准 [mV]
tau_syn      = 0.5;     % 突触时间常数 [ms]

% expLIF 额外参数（可选）
DeltaT = 5.0;   % [mV]
VT     = -50.0; % [mV]


%% ===================== 3. 连接参数（Python: config.py -> connection_params）=====================

% 相对抑制性突触强度
g_inh    = -16.0;   % 通用抑制
g_H      = -2.0;    % H1 类型
g_V      = -2.0;    % V 类型
g_P      = -2.0;    % P 类型
g_S      = -2.0;    % S 类型

% 突触权重参数
PSP_e        = 0.15;   % 兴奋性皮层内突触权重 [mV]
PSP_e_23_4   = 0.30;   % L2/3->L4 特化权重 [mV]
PSP_e_5_h1   = 0.15;   % L5->H1 特化权重 [mV]
PSP_ext      = 0.15;   % 外部输入突触权重 [mV]

% 权重相对标准差
PSC_rel_sd_normal   = 0.1;   % 正态分布
PSC_rel_sd_lognormal = 3.0;  % 对数正态分布

% 皮层-皮层连接缩放因子
cc_weights_factor   = 1.0;   % chi
cc_weights_I_factor = 0.8;   % chi_I

% alpha_norm: 各群体归一化因子
alpha_norm_keys = {'H1','E23','S23','V23','P23','E4','S4','V4','P4','E5','S5','V5','P5','E6','S6','V6','P6'};
alpha_norm_vals = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
alpha_norm = containers.Map(alpha_norm_keys, alpha_norm_vals);

% beta_norm: 各群体 beta 归一化因子
beta_norm_keys = {'H1','E23','S23','P23','V23','E4','S4','P4','V4','E5','S5','P5','V5','E6','S6','P6','V6'};
beta_norm_vals = [3.9,0.8,0.5,0.5,1.0,0.415,0.85,0.8,0.46,0.95,0.6,1.09,1.2,1.12,0.9,0.42,0.5];
beta_norm = containers.Map(beta_norm_keys, beta_norm_vals);


%% ===================== 4. 刺激参数（Python: config.py -> stim）=====================
stim_info = struct();
stim_info.V1.E23 = 10.0;   % 对 V1 区 E23 群体的刺激强度 [pA]

% 刺激时间窗口（由命令行参数控制，这里给默认值）
stim_start = 300;   % ms
stim_end   = 800;   % ms
duration   = 1000;  % ms（总仿真时长）
duration_timesteps = round(duration / DT_MS);


%% ===================== 5. 模型初始化（Python: GeNNModel(...)）=====================
% Python 代码:
%   model = GeNNModel("float", "GenCODE/" + model_name, ...)
%   model.dt = 0.1;                                          % 时间步长
%   model.fuse_postsynaptic_models = true;                   % 融合突触后模型
%   model.default_narrow_sparse_ind_enabled = true;           % 窄稀疏索引
%   model.timing_enabled = true;                              % 计时
%   model.default_var_location = VarLocation.HOST_DEVICE;     % 变量位置
%   model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE;

% MATLAB 等价记录：
model = struct();
model.precision = 'float';          % 数值精度
model.dt = DT_MS;                   % 时间步长 0.1 ms
model.fuse_postsynaptic = true;     % 融合突触后模型（提升性能）
model.narrow_sparse_ind = true;     % 窄稀疏索引（节省内存）
model.timing_enabled = true;        % 计时分析
model.var_location = 'HOST_DEVICE'; % 变量双端存储
model.sparse_conn_location = 'HOST_DEVICE';

fprintf('\n=== 模型配置 ===\n');
fprintf('精度: %s\n', model.precision);
fprintf('时间步长: %.1f ms\n', model.dt);
fprintf('融合突触后模型: %d\n', model.fuse_postsynaptic);
fprintf('窄稀疏索引: %d\n', model.narrow_sparse_ind);


%% ===================== 6. 突触后模型（Python: init_postsynaptic）=====================
% Python 代码:
%   exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5});

% ExpCurr: 指数衰减电流突触
%   I_syn(t) = g * exp(-t/tau)
ps_model = struct();
ps_model.type = 'ExpCurr';      % 指数衰减电流
ps_model.tau  = 0.5;            % 衰减时间常数 [ms]


%% ===================== 7. 电流源模型（Python: create_current_source_model）=====================
% Python 代码:
%   trigger_pulse_model = pygenn.create_current_source_model(
%       "trigger_pulse",
%       params=["start_time","end_time","magnitude"],
%       injection_code=
%       """
%       if (t >= start_time && t < end_time) {
%           injectCurrent(magnitude);
%       }
%       """
%   )

% TriggerPulse: 在时间窗口内注入恒定电流
cs_trigger = struct();
cs_trigger.name = 'TriggerPulse';
cs_trigger.params = {'start_time', 'end_time', 'magnitude'};
cs_trigger.description = '在 [start_time, end_time) 内注入 magnitude 电流';

% PoissonExp: 泊松输入（背景噪声）
%   参数: weight, tauSyn, rate
cs_poisson = struct();
cs_poisson.name = 'PoissonExp';
cs_poisson.params = {'weight', 'tauSyn', 'rate'};
cs_poisson.description = '泊松分布外部输入';


%% ===================== 8. 神经元群体创建 ======================
% Python 循环:
%   for area in area_list:
%       for pop in pop_list:
%           popName = area + pop
%           params["C"]   = Cm[pop] / 1000.0          % [nF]
%           params["TauM"] = Cm[pop] / gL[pop]         % [ms]
%           params["Vrest"] = Vrest[pop]                % [mV]
%           params["Vreset"] = Vrest[pop] - 10.0        % [mV]
%           params["Vthresh"] = Vth[pop]                % [mV]
%           params["TauRefrac"] = tref[pop]             % [ms]
%           params["Ioffset"] = input_current[pop] / 1000.0  % [nA]
%           neuron_pop = model.add_neuron_population(popName, pop_size, "LIF", params, lif_init)

% 记录所有创建的神经元群体
neuron_populations = struct();  % 结构体: neuron_populations.(area).(pop)

total_neurons   = 0;
neuron_group    = 0;

fprintf('\n=== 神经元群体参数（以 V1 区为例）===\n');
fprintf('%-12s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Population', 'C(nF)', 'TauM(ms)', 'Vrest(mV)', 'Vreset(mV)', 'Vth(mV)', 'Tref(ms)');

% 注意：实际运行时 NeuronNumber 从 JSON 加载
% 这里以 V1 区为例，展示参数计算过程
for p = 1:length(pop_list)
    pop = pop_list{p};
    popName = ['V1_', pop];   % 示例：仅 V1 区
    
    % 查表获取参数
    C_val   = Cm(pop) / 1000.0;           % Cm [pF] -> C [nF]
    TauM    = Cm(pop) / gL(pop);           % tau_m = C/gL [ms]
    V_rest  = Vrest(pop);                  % 静息电位
    V_reset = Vrest(pop) - 10.0;           % 重置电位
    V_thresh = Vth(pop);                   % 阈值
    T_ref   = tref(pop);                   % 不应期
    I_offset = input_current(pop) / 1000.0; % DC 电流 [nA]
    
    % LIF 初始条件
    % lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}),
    %             "RefracTime": T_ref}
    V_init_mean   = -150.0;   % [mV]（初始膜电位正态分布均值）
    V_init_sd     = 50.0;     % [mV]（标准差）
    
    fprintf('%-12s %-10.4f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
        popName, C_val, TauM, V_rest, V_reset, V_thresh, T_ref);
    
    % 记录（实际运行时 pop_size = NeuronNumber(area, pop) > 0 才创建）
    neuron_populations.V1.(pop) = struct(...
        'C',         C_val, ...
        'TauM',      TauM, ...
        'Vrest',     V_rest, ...
        'Vreset',    V_reset, ...
        'Vthresh',   V_thresh, ...
        'TauRefrac', T_ref, ...
        'Ioffset',   I_offset, ...
        'V_init_mean', V_init_mean, ...
        'V_init_sd',   V_init_sd);
end


%% ===================== 9. 突触群体创建 ======================
% Python 循环:
%   for areaTar, areaSrc in product(area_list, area_list):
%       for popTar, popSrc in product(pop_list, pop_list):
%           wAve = SynapsesWeightMean[areaTar][popTar][areaSrc][popSrc]/1000.0 * factor
%           wSd  = SynapsesWeightSd[areaTar][popTar][areaSrc][popSrc]/1000.0 * factor
%           synNum = SynapsesNumber[areaTar][popTar][areaSrc][popSrc]
%           meanDelay = delayMap[areaTar][popTar][areaSrc][popSrc]['ave']
%           delay_sd  = delayMap[areaTar][popTar][areaSrc][popSrc]['sd']
%           max_d     = delayMap[areaTar][popTar][areaSrc][popSrc]['max']
%           if synNum > 0:
%               connect_params = {"num": synNum}
%               d_dist = {"mean": meanDelay, "sd": delay_sd, "min": 0, "max": max_d}
%               if popSrc.startswith("E"):       % 兴奋性
%                   w_dist = {"mean": wAve, "sd": wSd, "min": 0, "max": +inf}
%               else:                             % 抑制性
%                   w_dist = {"mean": wAve, "sd": wSd, "min": -inf, "max": 0}
%
%               static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {},
%                   {"g": init_var("NormalClipped", w_dist),
%                    "d": init_var("NormalClippedDelay", d_dist)})
%
%               syn_pop = model.add_synapse_population(synName, matrix_type,
%                       neuron_populations[areaSrc][popSrc],
%                       neuron_populations[areaTar][popTar],
%                       static_synapse_init, exp_curr_init,
%                       init_sparse_connectivity("FixedNumberTotalWithReplacement", connect_params))
%
%               syn_pop.max_dendritic_delay_timesteps = round(max_d / DT_MS)

% 突触群体记录结构
synapse_populations = struct();  % synapse_populations.(tar_area).(tar_pop).(src_area).(src_pop)

fprintf('\n=== 突触连接参数说明 ===\n');
fprintf('连接类型: StaticPulseDendriticDelay (静态脉冲 + 树突延迟)\n');
fprintf('突触后模型: ExpCurr (指数衰减电流, tau=%.1f ms)\n', ps_model.tau);
fprintf('权重分布: NormalClipped (截断正态分布)\n');
fprintf('延迟分布: NormalClippedDelay (截断正态延迟分布)\n');
fprintf('稀疏连接: FixedNumberTotalWithReplacement (固定总数有放回抽样)\n');
fprintf('矩阵类型: SPARSE 或 PROCEDURAL（取决于 --SPARSE 参数）\n');

% 以 V1 区内 E4->E23 连接为例展示参数计算（实际从 JSON 加载）
fprintf('\n=== 示例：V1 区内连接 V1_E4 -> V1_E23 ===\n');

% 假设数据（实际从 SynapsesNumber, SynapsesWeightMean, delayMap 中取值）
example_synNum    = 1000000;     % 突触总数
example_wAve      = 0.15;        % 平均权重 [nA]
example_wSd       = 0.015;       % 权重标准差
example_meanDelay = 1.5;         % 平均延迟 [ms]
example_delaySd   = 0.5;         % 延迟标准差
example_maxDelay  = 3.0;         % 最大延迟 [ms]
example_factor    = 1.0;         % 缩放因子（同区用 Ind 归一化）

wAve_scaled = example_wAve / 1000.0 * example_factor;   % [uS]
wSd_scaled  = example_wSd  / 1000.0 * example_factor;

% 兴奋性连接：权重 >= 0
w_dist_exc = struct('mean', wAve_scaled, 'sd', wSd_scaled, 'min', 0.0, 'max', realmax('single'));
% 抑制性连接：权重 <= 0
w_dist_inh = struct('mean', wAve_scaled, 'sd', wSd_scaled, 'min', -realmax('single'), 'max', 0.0);

% 延迟分布
d_dist = struct('mean', example_meanDelay, 'sd', example_delaySd, 'min', 0.0, 'max', example_maxDelay);

% 连接参数
connect_params = struct('num', example_synNum);

% 树突延迟步数
max_delay_steps = round(example_maxDelay / DT_MS);

fprintf('  突触数量:          %d\n', example_synNum);
fprintf('  权重 (uS):         %.4f +/- %.4f\n', wAve_scaled, wSd_scaled);
fprintf('  延迟 (ms):         %.2f +/- %.2f (max=%.2f)\n', example_meanDelay, example_delaySd, example_maxDelay);
fprintf('  最大延迟步数:       %d\n', max_delay_steps);
fprintf('  连接建立方式:       FixedNumberTotalWithReplacement\n');
fprintf('  权重类型:           %s\n', 'NormalClipped (兴奋性: [0,+inf), 抑制性: (-inf,0])');


%% ===================== 10. 模型构建 & 加载（Python: build / load）=====================
% Python 代码:
%   model.build()   % 代码生成 + 编译
%   model.load()    % GPU 显存分配
%
% 编译时间、加载时间用于性能分析

fprintf('\n=== 构建和加载 ===\n');
fprintf('build():  GeNN 代码生成 + CUDA 编译\n');
fprintf('load():   GPU 显存分配 + 初始化\n');


%% ===================== 11. 仿真循环（Python: while model.t < duration）=====================
% Python 代码:
%   while model.t < duration:
%       model.step_time()      % 推进一个时间步 (0.1 ms)
%       model.pull_recording_buffers_from_device()  % 拉取 spike 数据
%       record_spike(...)

fprintf('\n=== 仿真循环 ===\n');
fprintf('总仿真时长:   %d ms\n', duration);
fprintf('总时间步数:   %d\n', duration_timesteps);
fprintf('每步操作:\n');
fprintf('  1. model.step_time() — 更新神经元 + 突触\n');
fprintf('  2. pull_recording_buffers_from_device() — 拉取 spike\n');
fprintf('  3. record_spike() — 记录 spike 数据\n');


%% ===================== 12. 性能计时汇总 ======================
% Python 代码（仿真结束后打印）:
%   model.init_time            % 初始化时间
%   model.init_sparse_time     % 稀疏矩阵初始化时间
%   model.neuron_update_time   % 神经元更新时间
%   model.presynaptic_update_time  % 突触前更新时间

fprintf('\n=== 性能计时字段 ===\n');
fprintf('  init_time:                变量初始化耗时\n');
fprintf('  init_sparse_time:         稀疏连接初始化耗时\n');
fprintf('  neuron_update_time:       神经元状态更新总耗时\n');
fprintf('  presynaptic_update_time:  突触前更新总耗时\n');


%% ===================== 13. 模型结构汇总 ======================
fprintf('\n%s\n', repmat('=', 1, 50));
fprintf('           模 型 结 构 汇 总\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('脑区数量:            %d\n', num_areas);
fprintf('群体类型数量:         %d\n', num_pops);
fprintf('总神经元群体数:       %d x %d = %d (最大值)\n', num_areas, num_pops, num_areas*num_pops);
fprintf('神经元模型:           LIF (Leaky Integrate-and-Fire)\n');
fprintf('突触后模型:           ExpCurr (tau=%.1f ms)\n', ps_model.tau);
fprintf('突触权重更新:         StaticPulseDendriticDelay\n');
fprintf('延迟类型:             树突延迟 (dendritic delay)\n');
fprintf('连接方式:             FixedNumberTotalWithReplacement\n');
fprintf('连接矩阵类型:         PROCEDURAL / SPARSE\n');
fprintf('电流源:               TriggerPulse (刺激) + PoissonExp (背景)\n');
fprintf('仿真精度:             %s\n', model.precision);
fprintf('时间步长:             %.1f ms\n', model.dt);
fprintf('脑区列表:\n');
for i = 1:num_areas
    fprintf('  %-6s', area_list{i});
    if mod(i, 8) == 0, fprintf('\n'); end
end
fprintf('\n');
fprintf('群体类型:\n');
for i = 1:num_pops
    fprintf('  %-4s', pop_list{i});
    if mod(i, 8) == 0, fprintf('\n'); end
end
fprintf('\n%s\n', repmat('=', 1, 50));

fprintf('\n转换完成。本脚本记录了 CustomModel.py 的完整建模逻辑。\n');
