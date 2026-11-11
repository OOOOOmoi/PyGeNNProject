#!/bin/bash
# V1 单区：第四层 SST<->VIP 互抑权重二维联合扫描
# sst2vip (S4->V4) ∈ [0,1] 步长0.05 (21值)，vip2sst (V4->S4) ∈ [0,1] 步长0.05 (21值)
# 共 21 × 21 = 441 组合
# 用于定位 winner-take-all 相变来源：关断/缩放任一条互抑通路观察相变是否消失/移动
# 编译目录按结构参数(getCompileName)解耦，运行时权重缩放参数复用同一编译结果
# 每张卡串行跑自己的组合队列(避免同卡并发编译冲突)，卡间并行
# 用法: bash runner_v1_s2v_v2s_2d.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")/.."

DURATION=500
EXT_SCALE=0.02
GPUS=(0 1 2 3 4 5 6 7 8 9)   # 当前空闲卡（nvidia-smi 全空闲）

# sst2vip / vip2sst: 0.0 ~ 1.0 步长 0.05，共 21 个
S2V_LIST=$(python3 -c "print(' '.join(f'{0.05*i:.2f}' for i in range(21)))")
V2S_LIST=$(python3 -c "print(' '.join(f'{0.05*i:.2f}' for i in range(21)))")

mkdir -p log

# 生成所有组合
COMBOS=()
for s2v in $S2V_LIST; do
    for v2s in $V2S_LIST; do
        COMBOS+=("$s2v $v2s")
    done
done
N=${#COMBOS[@]}
echo "总组合数: $N (21 sst2vip × 21 vip2sst)"

# 把组合轮流分配到各卡（每卡串行处理自己的队列）
NGPU=${#GPUS[@]}
for gi in $(seq 0 $((NGPU-1))); do
    gpu=${GPUS[$gi]}
    (
        ci=$gi
        while [ $ci -lt $N ]; do
            read s2v v2s <<< "${COMBOS[$ci]}"
            TAG="d${DURATION}_ext${EXT_SCALE}_s2v${s2v}_v2s${v2s}"
            RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
            echo "[GPU $gpu] sst2vip=$s2v vip2sst=$v2s"
            python CustomModel_MPI.py --duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --sst2vip-scale $s2v --vip2sst-scale $v2s > "$RUN_LOG" 2>&1
            ci=$((ci + NGPU))
        done
    ) &
done
wait
echo "全部 ${N} 个组合完成"
