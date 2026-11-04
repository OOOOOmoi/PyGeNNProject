#!/bin/bash
# V1 单区 VIP × SST 高分辨率二维网格扫描（0.05 步长，20×20=400 组合）
# 每张卡同时跑 2 个组合（编译目录已按 model_name 隔离）
# 10 卡 × 每批 20 组合 = 20 批
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
GPUS=(0 1 2 3 4 5 6 7 8 9)

mkdir -p log

# 生成 0.05~1.00 共 20 个值
VIP_LIST="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"
SST_LIST="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"

# 生成所有组合
COMBOS=()
for v in $VIP_LIST; do
    for s in $SST_LIST; do
        COMBOS+=("$v $s")
    done
done

N=${#COMBOS[@]}
echo "总组合数: $N"

BATCH=20   # 每批 20 个（每卡 2 个）
idx=0
batch_no=0
for combo in "${COMBOS[@]}"; do
    read v s <<< "$combo"
    gpu=${GPUS[$((idx % 10))]}      # 10 张卡轮询，每批每卡 2 个
    TAG="d${DURATION}_ext${EXT_SCALE}_vipext${v}_sstext${s}"
    RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
    ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $EXT_SCALE --vip-ext-scale $v --sst-ext-scale $s"
    nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
    idx=$((idx+1))
    if [ $((idx % BATCH)) -eq 0 ]; then
        batch_no=$((batch_no+1))
        echo "--- 第 $batch_no 批 (20 个) 已启动，等待完成 ---"
        wait
        echo "--- 第 $batch_no 批完成 ---"
    fi
done
wait
echo "全部 ${N} 个高分辨率组合任务完成"
