#!/bin/bash

find /home/yangjinhao/PyGeNNProject/EIBalance/output -name "*.png" -type f -delete
#!/bin/bash

TARGET_DIR="./CODE"

if [ -d "$TARGET_DIR" ]; then
    echo "清空目录 $TARGET_DIR 中的所有内容..."
    rm -rf "$TARGET_DIR"/*
else
    echo "目录 $TARGET_DIR 不存在，正在创建..."
    mkdir -p "$TARGET_DIR"
fi

# 捕获 Ctrl+C 或 kill 信号，杀死所有子进程
trap "echo 'Stopping runner...'; kill 0; exit" SIGINT SIGTERM

values=$(seq 10 10 100)
ngpu=8
process_per_gpu=2
N=process_per_gpu*ngpu
count=0
for w1 in $values; do
for w2 in $values; do
for w3 in $values; do
for w4 in $values; do
  gpu_id=$((count % ngpu))
  echo "Launching task $count on GPU $gpu_id"
  
  python EIBalance.py \
    --wEE $w1 --wEI $w2 --wIE $w3 --wII $w4  \
    --device $gpu_id &

  if (( (count+1) % N == 0 )); then
    wait
  fi
  ((count++))
done
done
done
done

wait
