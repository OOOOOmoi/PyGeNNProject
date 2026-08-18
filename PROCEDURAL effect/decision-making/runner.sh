#!/bin/bash
# Wang 2002 Decision-Making Model — MPI Runner
# Usage: bash runner.sh [coherence] [seed] [num_workers] [batch_steps]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0

COH=${1:-51.2}
SEED=${2:-4}
NW=${3:-4}
BS=${4:-20}

cd "$(dirname "$0")"
mkdir -p GenCODE output log

echo "===== Wang 2002 MPI | coh=${COH} seed=${SEED} workers=${NW} batch=${BS} | $(date) ====="
python3 wang2002_mpi.py \
    --duration 2000 \
    --coh ${COH} \
    --seed ${SEED} \
    --num-workers ${NW} \
    --gpu-ids 0 1 2 3 \
    --batch-steps ${BS}
echo "===== DONE | $(date) ====="
