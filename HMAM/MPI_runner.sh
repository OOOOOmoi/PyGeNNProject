#!/bin/bash

find /home/yangjinhao/PyGeNNProject/HMAM/output -name "*.png" -type f -delete
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

values1=$(seq 0 0.1 0)
values2=$(seq 0.1 0.1 1)
values3=$(seq 0.1 0.1 1)
values4=$(seq 0 0.1 0)
ngpu=8
count=0

for w1 in $values1; do
for w2 in $values2; do
for w3 in $values3; do
for w4 in $values4; do
  gpu_id=$((count % ngpu))
  echo "Launching task $count on GPU $gpu_id"
  
  python HMAM.py \
    --wEE $w1 --wEI $w2 --wIE $w3 --wII $w4  \
    --duration 3000 \
    --SPARSE \
    --device $gpu_id &

  if (( (count+1) % ngpu == 0 )); then
    wait
  fi
  ((count++))
done
done
done
done
wait
