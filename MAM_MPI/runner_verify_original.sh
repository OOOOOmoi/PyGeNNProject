#!/bin/bash
# V1 单区验证：原始 Allen 连接概率（custom_Data_Model_3396.json 作对照）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")"
DURATION=${1:-500}
ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids 1"
mkdir -p log
RUN_LOG="log/run_verify_original_$(date +%H%M%S).out"
nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$RUN_LOG"
echo "ARGS=$ARGS"
