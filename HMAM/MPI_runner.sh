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
values1=$(seq 0.5 0.1 0.5)
values2=$(seq 10 1 10)
values3=$(seq 3 1 3)
values4=$(seq 7 0.1 9)
values5=$(seq 0 1 67)
ngpu=8          # GPU 数量
per_gpu=2       # 每个 GPU 上允许的最大并行进程数
count=0

# 任务管理
running=()

for w1 in $values1; do
for w2 in $values2; do
for w3 in $values3; do
for w4 in $values4; do
for w5 in $values5; do
  gpu_id=$((count % ngpu))
  echo "Launching task $count on GPU $gpu_id"

  ARGS="--duration 3000"
  ARGS="$ARGS --buffer"
  ARGS="$ARGS --buffer-size 1000"
  # ARGS="$ARGS --SPARSE"
  ARGS="$ARGS --wEE $w1 --wEI $w2 --wIE $w3 --wII $w4"
  ARGS="$ARGS --AreaIdx $w5"
  ARGS="$ARGS --device $gpu_id"
  ARGS="$ARGS --poisson"
  # ARGS="$ARGS --scaleSyn 0.01"
  # ARGS="$ARGS --inSyn"

  python HMAM.py $ARGS &

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
done
wait
