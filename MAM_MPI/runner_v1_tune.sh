#!/bin/bash
# V1 单脑区调参专用脚本 (外部输入/突触权重/抑制强度缩放扫描)
# 用法: bash runner_v1_tune.sh <duration> <ext-scale> <syn-scale> <inh-scale>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")"

DURATION=${1:-500}
EXT_SCALE=${2:-1.0}
SYN_SCALE=${3:-1.0}
INH_SCALE=${4:-1.0}

ARGS="--duration $DURATION"
ARGS="$ARGS --AreaNum 1"
ARGS="$ARGS --save-spike"
ARGS="$ARGS --gpu-ids 0"
ARGS="$ARGS --ext-scale $EXT_SCALE"
ARGS="$ARGS --syn-scale $SYN_SCALE"
ARGS="$ARGS --inh-scale $INH_SCALE"

mkdir -p log
RUN_LOG="log/run_v1_d${DURATION}_ext${EXT_SCALE}_syn${SYN_SCALE}_inh${INH_SCALE}_$(date +%H%M%S).out"
echo "ARGS: $ARGS"
echo "LOG: $RUN_LOG"
nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
echo "PID=$!"
