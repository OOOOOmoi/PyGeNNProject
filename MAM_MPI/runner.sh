#!/bin/bash
# MAM_MPI 32 区完整模型 — 500ms 仿真，使用空闲的 GPU 0/5/6
# 保留 spike CSV 输出 (save_spike() 已取消注释)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52

# CUDA env vars 必须显式 export, 因为 conda env vars 不会通过 Process() 继承给子进程
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}

set -e
cd "$(dirname "$0")"

# 清空旧的 spike / volt / inSyn CSV 避免污染本次输出
find /home/yangjinhao/PyGeNNProject/MAM_MPI/output -name "*.csv" -type f -delete

ARGS="--duration 500"
ARGS="$ARGS --AreaNum 32"
ARGS="$ARGS --save-spike"          # 触发 CustomModel_MPI.py 末尾的 save_spike() 调用
ARGS="$ARGS --gpu-ids 0 5 6"       # 仅使用空闲的 0/5/6 号 GPU

# 可选刺激 / 缩放参数 (按需取消注释)
# ARGS="$ARGS --scale 5"
# ARGS="$ARGS --stim-start 100"
# ARGS="$ARGS --stim-end 200"
# ARGS="$ARGS --inSyn"

mkdir -p log
RUN_LOG="log/run_$(date +%Y%m%d_%H%M%S)_save_spike.out"
nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$RUN_LOG"
