#!/bin/bash
# V1 单区：VIP gL 增益扫描实验
# 目的：增大 VIP 的 gL（降低 f-I 增益），观察 VIP<->SST 相变是否变软
# 固定 ext-scale=0.02，扫 vip-gl-scale x vipext 二维网格
# 用法: bash runner_v1_vipgl_scan.sh
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
GL_LIST="1.0 2.0 4.0 8.0"
VIPEXT_LIST="0.25 0.30 0.325 0.35 0.40"
GPUS=(0 1 2 7 8 9)
BATCH=6

mkdir -p log

read -ra GL_ARR <<< "$GL_LIST"
read -ra VIP_ARR <<< "$VIPEXT_LIST"

idx=0
total=0
for gl in "${GL_ARR[@]}"; do
    for v in "${VIP_ARR[@]}"; do
        gpu=${GPUS[$((idx % BATCH))]}
        TAG="d${DURATION}_ext${EXT_SCALE}_vipgl${gl}_vipext${v}"
        RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
        ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-gl-scale $gl --vip-ext-scale $v"
        echo "[launch] vipgl=$gl vipext=$v -> GPU $gpu  LOG=$RUN_LOG"
        nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
        echo "  PID=$!"
        idx=$((idx+1))
        total=$((total+1))
        if [ $((idx % BATCH)) -eq 0 ]; then
            echo "--- 第 $((idx/BATCH)) 批 (6 个) 已启动，等待完成 ---"
            wait
            echo "--- 第 $((idx/BATCH)) 批完成 ---"
        fi
    done
done
wait
echo "全部 ${total} 个组合完成"
