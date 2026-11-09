#!/bin/bash
# V1 单区：0.01 步长细粒度二维网格扫描
# gL ∈ {1.0, 2.0}，vipext ∈ [0.20, 0.50] 步长0.01(31值)，sstext ∈ [0.05, 1.00] 步长0.01(96值)
# 共 2 × 31 × 96 = 5952 组合
# 编译目录已按结构参数(getCompileName)解耦，运行时缩放参数复用同一编译结果
# 每张卡串行跑自己的组合队列(避免同卡并发编译冲突)，卡间并行
# 用法: bash runner_v1_0p01_fine_2d.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")"

DURATION=500
EXT_SCALE=0.02
GPUS=(0 1 2 3 4 5 6 7 8 9)   # 10 张空闲卡
GL_LIST="1.0 2.0"

# vipext: 0.20~0.50 步长 0.01，共 31 个
VIP_LIST="0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.50"
# sstext: 0.05~1.00 步长 0.01，共 96 个
SST_LIST=$(python3 -c "print(' '.join(f'{0.05+0.01*i:.2f}' for i in range(96)))")

mkdir -p log

# 生成所有组合
COMBOS=()
for gl in $GL_LIST; do
    for v in $VIP_LIST; do
        for s in $SST_LIST; do
            COMBOS+=("$gl $v $s")
        done
    done
done
N=${#COMBOS[@]}
echo "总组合数: $N (2 gl × 31 vipext × 96 sstext)"

# 把组合轮流分配到各卡（每卡串行处理自己的队列）
NGPU=${#GPUS[@]}
for gi in $(seq 0 $((NGPU-1))); do
    gpu=${GPUS[$gi]}
    (
        ci=$gi
        while [ $ci -lt $N ]; do
            read gl v s <<< "${COMBOS[$ci]}"
            TAG="d${DURATION}_ext${EXT_SCALE}_vipgl${gl}_vipext${v}_sstext${s}"
            RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
            echo "[GPU $gpu] vipgl=$gl vipext=$v sstext=$s"
            python CustomModel_MPI.py --duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-gl-scale $gl --vip-ext-scale $v --sst-ext-scale $s > "$RUN_LOG" 2>&1
            ci=$((ci + NGPU))
        done
    ) &
done
wait
echo "全部 ${N} 个组合完成"
