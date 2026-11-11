#!/bin/bash
# 验证新编译目录逻辑：不同运行时参数是否复用同一编译结果
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
cd /home/yangjinhao/PyGeNNProject/MAM_MPI

echo "=== 组合1 (vipgl1.5 vipext0.5, 首次编译) ==="
time python CustomModel_MPI.py --duration 500 --AreaNum 1 --save-spike --gpu-ids 0 --ext-scale 0.02 --vip-gl-scale 1.5 --vip-ext-scale 0.5 > log/test_reuse_1.out 2>&1
echo "exit=$?"

echo "=== 组合2 (vipgl1.5 vipext0.6, 应复用编译) ==="
time python CustomModel_MPI.py --duration 500 --AreaNum 1 --save-spike --gpu-ids 0 --ext-scale 0.02 --vip-gl-scale 1.5 --vip-ext-scale 0.6 > log/test_reuse_2.out 2>&1
echo "exit=$?"

echo "=== 编译目录 ==="
ls -d GenCODE/worker0_on_device0_0.5s_AreaNum1* 2>/dev/null
echo "=== 组合1 日志尾部 ==="
tail -5 log/test_reuse_1.out
