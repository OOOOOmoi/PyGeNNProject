#!/bin/bash
# V1 单区 同步降低 VIP + SST 外部输入（双降扫描）
# 固定 ext-scale=0.02，vipext = sstext = scale 同步下降
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
SCALE_LIST="0.7 0.5 0.4 0.3 0.2 0.1 0.05"
GPUS=(0 1 2 7 8 9)

mkdir -p log

read -ra SCALE_ARR <<< "$SCALE_LIST"
N=${#SCALE_ARR[@]}
BATCH=6
idx=0
for s in "${SCALE_ARR[@]}"; do
    gpu=${GPUS[$((idx % BATCH))]}
    TAG="d${DURATION}_ext${EXT_SCALE}_vipext${s}_sstext${s}"
    RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
    ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-ext-scale $s --sst-ext-scale $s"
    echo "[launch] scale=$s -> GPU $gpu  LOG=$RUN_LOG"
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
echo "全部 ${N} 个双降任务完成"
