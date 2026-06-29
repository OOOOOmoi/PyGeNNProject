#!/bin/bash
set -e
PY=/home/yangjinhao/miniconda3/envs/pygenn52/bin/python
export CUDA_PATH=/usr/local/cuda-12.0
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export NUMBA_CUDA_LIBDEVICE=/usr/local/cuda-12.0/nvvm/libdevice
cd ~/PyGeNNProject
mkdir -p GenCODE
echo "===== GPU=2 | $(date) ====="
$PY MAM_MPI/CustomModel_MPI.py --AreaNum 32 --duration 10000 --gpu-ids 0 3
echo "===== GPU=3 | $(date) ====="
$PY MAM_MPI/CustomModel_MPI.py --AreaNum 32 --duration 10000 --gpu-ids 0 3 4
echo "===== GPU=4 | $(date) ====="
$PY MAM_MPI/CustomModel_MPI.py --AreaNum 32 --duration 10000 --gpu-ids 0 3 4 6
echo "===== GPU=5 | $(date) ====="
$PY MAM_MPI/CustomModel_MPI.py --AreaNum 32 --duration 10000 --gpu-ids 0 3 4 6 8
echo "===== GPU=6 | $(date) ====="
$PY MAM_MPI/CustomModel_MPI.py --AreaNum 32 --duration 10000 --gpu-ids 0 3 4 6 8 9
echo "===== ALL DONE | $(date) ====="