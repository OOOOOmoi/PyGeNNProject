#!/bin/bash
# V1 单区：VIP gL × vipext 二维网格扫描（gL: 1.0~2.0 步长0.05 共21组 × vipext: 0.05~1.0 步长0.05 共20组 = 420组合）
# 目的：定位相变"硬→软"临界 gL，并绘制 17 群体放电率热力图
# 编译目录已按结构参数(getCompileName)解耦，运行时缩放参数复用同一编译结果
# 每张卡串行跑自己的组合队列(避免同卡并发编译冲突)，卡间并行
# 用法: bash runner_v1_vipgl_2d_grid.sh
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
GPUS=(0 1 2 7 8 9)   # 6 张空闲卡

# gL: 1.00~2.00 步长 0.05，共 21 个
GL_LIST="1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00"
# vipext: 0.05~1.00 步长 0.05，共 20 个
VIP_LIST="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"

mkdir -p log

# 生成所有组合
COMBOS=()
for gl in $GL_LIST; do
    for v in $VIP_LIST; do
        COMBOS+=("$gl $v")
    done
done
N=${#COMBOS[@]}
echo "总组合数: $N (21 gl × 20 vipext)"

# 把组合轮流分配到各卡（每卡串行处理自己的队列）
NGPU=${#GPUS[@]}
for gi in $(seq 0 $((NGPU-1))); do
    gpu=${GPUS[$gi]}
    (
        ci=$gi
        while [ $ci -lt $N ]; do
            read gl v <<< "${COMBOS[$ci]}"
            TAG="d${DURATION}_ext${EXT_SCALE}_vipgl${gl}_vipext${v}"
            RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
            echo "[GPU $gpu] vipgl=$gl vipext=$v"
            python CustomModel_MPI.py --duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-gl-scale $gl --vip-ext-scale $v > "$RUN_LOG" 2>&1
            ci=$((ci + NGPU))
        done
    ) &
done
wait
echo "全部 ${N} 个组合完成"
