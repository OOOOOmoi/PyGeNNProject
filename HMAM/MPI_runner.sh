#!/bin/bash

find /home/yangjinhao/PyGenn/HMAM/output -name "*.png" -type f -delete
#!/bin/bash

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

values=$(seq 0 5 100)
ngpu=10
count=0

for w in $values; do
  gpu_id=$((count % ngpu))
  echo "Launching task $count on GPU $gpu_id"
  
  python HMAM.py \
    --wEE 0 --wEI 50 --wIE $w --wII 0  \
    --device $gpu_id &

  if (( (count+1) % ngpu == 0 )); then
    wait
  fi
  ((count++))
done

wait
