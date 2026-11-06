#!/bin/bash
# V1 单区 Vreset=Vrest 基线验证 (duration=500ms, ext-scale=0.02, GPU 0)
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
GPU=0

ARGS="--duration $DURATION"
ARGS="$ARGS --AreaNum 1"
ARGS="$ARGS --save-spike"
ARGS="$ARGS --gpu-ids $GPU"
ARGS="$ARGS --ext-scale $EXT_SCALE"

mkdir -p log
TAG="d${DURATION}_ext${EXT_SCALE}_vresetVrest"
RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
echo "ARGS: $ARGS"
echo "LOG: $RUN_LOG"
python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1
echo "DONE, exit=$?"
echo "LOG=$RUN_LOG"
