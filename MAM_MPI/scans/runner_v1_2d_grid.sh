#!/bin/bash
# V1 单区 VIP × SST 外部输入二维网格扫描（系统化）
# 固定 ext-scale=0.02，vipext × sstext 完整网格
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
VIP_LIST="0.05 0.1 0.2 0.3 0.5 0.7 1.0"
SST_LIST="0.05 0.1 0.2 0.3 0.5 0.7 1.0"
GPUS=(0 1 2 3 4 5 6 7 8 9)

mkdir -p log

COMBOS=()
for v in $VIP_LIST; do
    for s in $SST_LIST; do
        COMBOS+=("$v $s")
    done
done

N=${#COMBOS[@]}
BATCH=10
idx=0
for combo in "${COMBOS[@]}"; do
    read v s <<< "$combo"
    gpu=${GPUS[$((idx % BATCH))]}
    TAG="d${DURATION}_ext${EXT_SCALE}_vipext${v}_sstext${s}"
    RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
    ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-ext-scale $v --sst-ext-scale $s"
    echo "[launch] vipext=$v sstext=$s -> GPU $gpu  LOG=$RUN_LOG"
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
echo "全部 ${N} 个二维网格任务完成"
