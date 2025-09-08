#!/bin/bash

# 清理旧文件
find /home/yangjinhao/PyGeNNProject/HMAM/output -name "*.png" -type f -delete

TARGET_DIR="./HMAM_CODE"
if [ -d "$TARGET_DIR" ]; then
    echo "清空目录 $TARGET_DIR 中的所有内容..."
    rm -rf "$TARGET_DIR"/*
else
    echo "目录 $TARGET_DIR 不存在，正在创建..."
    mkdir -p "$TARGET_DIR"
fi

# 捕获 Ctrl+C 或 kill 信号，杀死所有子进程
trap "echo 'Stopping runner...'; kill 0; exit" SIGINT SIGTERM

# 参数范围
values1=$(seq 0.1 0.01 0.1)
values2=$(seq 1 0.1 10)
values3=$(seq 1 0.1 10)
values4=$(seq 1 0.1 10)

ngpu=8          # GPU 数量
per_gpu=5       # 每个 GPU 上允许的最大并行进程数
count=0

# 任务管理
running=()

for w1 in $values1; do
for w2 in $values2; do
for w3 in $values3; do
for w4 in $values4; do
  gpu_id=$((count % ngpu))
  echo "Launching task $count on GPU $gpu_id"

  python HMAM.py \
    --wEE $w1 --wEI $w2 --wIE $w3 --wII $w4 \
    --duration 3000 \
    --SPARSE \
    --device $gpu_id &

  pid=$!
  running+=($pid)

  # 控制并发量（每个 GPU 最多 per_gpu 个进程）
  while true; do
    active=0
    for p in "${running[@]}"; do
      if kill -0 $p 2>/dev/null; then
        ((active++))
      fi
    done
    if (( active < ngpu * per_gpu )); then
      break
    fi
    sleep 2
  done

  ((count++))
done
done
done
done

wait
