#!/bin/bash
# V1 单区：gL=2.0 固定下，VIP-ext × SST-ext 二维网格扫描（20×20=400 组合）
# 目的：对比第一次(无 gL 缩放)的 vip-ext × sst-ext 网格，看 gL 放大 2 倍后相变/失衡结构如何变化
# 编译目录已按结构参数(getCompileName)解耦，运行时缩放参数复用同一编译结果
# 每张卡串行跑自己的组合队列(避免同卡并发编译冲突)，卡间并行
# 用法: bash runner_v1_vipgl2_2d.sh
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
VIPGL=2.0
GPUS=(0 1 2 7 8 9)   # 6 张空闲卡

# vip-ext: 0.05~1.00 步长 0.05，共 20 个
VIP_LIST="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"
# sst-ext: 0.05~1.00 步长 0.05，共 20 个
SST_LIST="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"

mkdir -p log

# 生成所有组合
COMBOS=()
for v in $VIP_LIST; do
    for s in $SST_LIST; do
        COMBOS+=("$v $s")
    done
done
N=${#COMBOS[@]}
echo "总组合数: $N (20 vip-ext × 20 sst-ext, gL=$VIPGL)"

# 把组合轮流分配到各卡（每卡串行处理自己的队列）
NGPU=${#GPUS[@]}
for gi in $(seq 0 $((NGPU-1))); do
    gpu=${GPUS[$gi]}
    (
        ci=$gi
        while [ $ci -lt $N ]; do
            read v s <<< "${COMBOS[$ci]}"
            TAG="d${DURATION}_ext${EXT_SCALE}_vipgl${VIPGL}_vipext${v}_sstext${s}"
            RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
            echo "[GPU $gpu] vipgl=$VIPGL vipext=$v sstext=$s"
            python CustomModel_MPI.py --duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-gl-scale $VIPGL --vip-ext-scale $v --sst-ext-scale $s > "$RUN_LOG" 2>&1
            ci=$((ci + NGPU))
        done
    ) &
done
wait
echo "全部 ${N} 个组合完成"
