#!/bin/bash
# V1 单区 ext-scale 并行扫描 — 6 张空闲卡 (0/1/2/7/8/9) 分批并行
# 每批 6 个组合同时跑，批内每卡一个进程，跑完一批 wait 后再跑下一批
# 用法: bash runner_v1_ext_scan.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0
export PATH=/home/yangjinhao/CUDA/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/home/yangjinhao/CUDA/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
set -e
cd "$(dirname "$0")"

DURATION=500
# ext-scale 扫描序列（几何覆盖 0.001 ~ 0.05 过渡区，12 个点）
EXT_LIST="0.002 0.004 0.006 0.008 0.010 0.015 0.020 0.030 0.050 0.001 0.003 0.005"
# 空闲 GPU 列表（每批 6 张）
GPUS=(0 1 2 7 8 9)

mkdir -p log

# 把 EXT_LIST 拆成数组
read -ra EXT_ARR <<< "$EXT_LIST"
N=${#EXT_ARR[@]}
BATCH=6

idx=0
for ext in "${EXT_ARR[@]}"; do
    gpu=${GPUS[$((idx % BATCH))]}
    TAG="d${DURATION}_ext${ext}"
    RUN_LOG="log/run_v1_${TAG}_$(date +%H%M%S).out"
    ARGS="--duration $DURATION --AreaNum 1 --save-spike --gpu-ids $gpu --ext-scale $ext"
    echo "[launch] ext=$ext -> GPU $gpu  LOG=$RUN_LOG"
    nohup python CustomModel_MPI.py $ARGS > "$RUN_LOG" 2>&1 &
    echo "  PID=$!"
    idx=$((idx+1))
    # 每满 6 个就等这批跑完
    if [ $((idx % BATCH)) -eq 0 ]; then
        echo "--- 第 $((idx/BATCH)) 批 (6 个) 已启动，等待完成 ---"
        wait
        echo "--- 第 $((idx/BATCH)) 批完成 ---"
    fi
done
# 收尾：等待可能不足一批的剩余任务
wait
echo "全部 ${N} 个任务完成，分布在 GPU ${GPUS[*]}"
