#!/bin/bash
# V1 单区：全局外部输入 ext-scale 精细扫描
# ext-scale ∈ [0, 0.1] 步长 0.001，共 101 个值
# 其它参数全部默认(vipext/sstext/gL=1.0)，观察外部输入对 17 群体放电率的影响
# 每张卡串行跑自己的组合队列(避免同卡并发编译冲突)，卡间并行
# 用法: bash runner_v1_ext_fine.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")/.."

DURATION=500
GPUS=(0 1 2 3 4 5 6 7 8 9)   # 当前空闲卡

# ext-scale: 0.000 ~ 0.100 步长 0.001，共 101 个
EXT_LIST=$(python3 -c "print(' '.join(f'{0.001*i:.3f}' for i in range(101)))")

mkdir -p log

# 生成数组
read -ra EXT_ARR <<< "$EXT_LIST"
N=${#EXT_ARR[@]}
echo "总组合数: $N (ext-scale 0.000~0.100 步长 0.001)"

# 把组合轮流分配到各卡（每卡串行处理自己的队列）
NGPU=${#GPUS[@]}
for gi in $(seq 0 $((NGPU-1))); do
    gpu=${GPUS[$gi]}
    (
        ci=$gi
        while [ $ci -lt $N ]; do
            ext=${EXT_ARR[$ci]}
            TAG="d${DURATION}_ext${ext}"
            RUN_LOG="log/run_v1_extfine_${TAG}_$(date +%H%M%S).out"
            echo "[GPU $gpu] ext-scale=$ext"
            python CustomModel_MPI.py --duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $ext > "$RUN_LOG" 2>&1
            ci=$((ci + NGPU))
        done
    ) &
done
wait
echo "全部 ${N} 个组合完成"
