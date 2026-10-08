#!/bin/bash
# Wang-Buzsaki 1996 ING 模型 GeNN 版运行脚本
# 用法: ./run.sh [wb1996_genn.py 的参数...]
#   ./run.sh                      # 默认 Msyn=100 全连接
#   ./run.sh --Msyn 60            # 稀疏连接
#   ./run.sh --Msyn 30 --tag desync

# CUDA 环境 (.bashrc 中的 export 行有粘连错误, 非交互 shell 不生效, 故在此显式设置)
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64

# 仅用 GPU 0/1 (其余被占用), 默认 GPU 0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=~/miniconda3/envs/pygenn52/bin/python
cd "$(dirname "$0")"
exec $PY wb1996_genn.py "$@"
