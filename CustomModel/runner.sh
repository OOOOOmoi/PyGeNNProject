# 判断是否传入 device 参数
if [ -z "$1" ]; then
    echo "用法: $0 <device_id>"
    exit 1
fi

DEVICE=$1
AreaNum=$2
# find /home/yangjinhao/PyGenn/CustomModel/output -name "*.png" -type f -delete
ARGS="--duration 1000"
ARGS="$ARGS --buffer"
ARGS="$ARGS --buffer-size 100"
ARGS="$ARGS --device $DEVICE"
ARGS="$ARGS --poisson"
ARGS="$ARGS --AreaNum $AreaNum"
# ARGS = "$ARGS --AreaIdx 0"
# ARGS = "$ARGS --SPARSE"
for i in {1..1}; do
    echo "========== Run $i =========="
    python CustomModel.py $ARGS
done
