find /home/yangjinhao/PyGenn/CustomModel/output -name "*.png" -type f -delete
#!/bin/bash

TARGET_DIR="./GenCODE"

if [ -d "$TARGET_DIR" ]; then
    echo "清空目录 $TARGET_DIR 中的所有内容..."
    rm -rf "$TARGET_DIR"/*
else
    echo "目录 $TARGET_DIR 不存在，正在创建..."
    mkdir -p "$TARGET_DIR"
fi

NUM_PROCESSES=10        # 要启动的进程数量
START_SCALE=0         # 初始 free-scale
SCALE_STEP=0.1          # 每个进程增加的 free-scale 步长

DURATION=1000
SCRIPT="CustomModel.py"

for ((i=0; i<NUM_PROCESSES; i++)); do
    GPU=$((i % 10)) 
    SCALE=$(awk "BEGIN { printf \"%.2f\", $START_SCALE + $i * $SCALE_STEP }")

    echo "Launching process $i on GPU $GPU with --free-scale $SCALE"

    python "$SCRIPT" --duration "$DURATION" --device "$GPU" --free-scale "$SCALE" &
done

echo "所有进程已启动。"
