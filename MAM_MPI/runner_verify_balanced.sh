#!/bin/bash
# V1 单区验证：balanced I->I 连接概率（E/I 对齐 Potjans）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")"
DURATION=${1:-500}
ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids 0"
ARGS="$ARGS --data-json model_info_schmidt_motif_diff_s/default_Data_Model__balanced_1.0mm2.json"
mkdir -p log
RUN_LOG="log/run_verify_balanced_$(date +%H%M%S).out"
nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$RUN_LOG"
echo "ARGS=$ARGS"
