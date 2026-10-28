#!/bin/bash
# V1 单区 vip-ext-scale 并行扫描 — 单独降低 VIP 神经元外部刺激强度
# 固定 ext-scale=0.02，扫 vip-ext-scale
# 用法: bash runner_v1_vipext_scan.sh
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
VIP_LIST="1.0 0.5 0.3 0.2 0.1 0.05"
GPUS=(0 1 2 7 8 9)

mkdir -p log

read -ra VIP_ARR <<< "$VIP_LIST"
N=${#VIP_ARR[@]}
BATCH=6
idx=0
for v in "${VIP_ARR[@]}"; do
    gpu=${GPUS[$((idx % BATCH))]}
    TAG="d${DURATION}_ext${EXT_SCALE}_vipext${v}"
    RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
    ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-ext-scale $v"
    echo "[launch] vipext=$v -> GPU $gpu  LOG=$RUN_LOG"
    nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
    echo "  PID=$!"
    idx=$((idx+1))
    if [ $((idx % BATCH)) -eq 0 ]; then
        echo "--- 第 $((idx/BATCH)) 批 (6 个) 已启动，等待完成 ---"
        wait
        echo "--- 第 $((idx/BATCH)) 批完成 ---"
    fi
done
wait
echo "全部 ${N} 个 vip-ext-scale 任务完成"
