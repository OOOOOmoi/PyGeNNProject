#!/bin/bash
# V1 单区 vip-ext-scale 细粒度扫描 — 定位 0.3~0.5 去抑制突变边界
# 固定 ext-scale=0.02
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
VIP_LIST="0.26 0.28 0.30 0.325 0.35 0.375 0.40 0.425 0.45 0.475 0.50 0.55"
GPUS=(0 1 2 3 4 5 6 7 8 9)   # 10 张卡全空

mkdir -p log

read -ra VIP_ARR <<< "$VIP_LIST"
N=${#VIP_ARR[@]}
BATCH=10
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
        echo "--- 第 $((idx/BATCH)) 批 ($BATCH 个) 已启动，等待完成 ---"
        wait
        echo "--- 第 $((idx/BATCH)) 批完成 ---"
    fi
done
wait
echo "全部 ${N} 个 vip-ext-scale 细粒度任务完成"
