# find /home/yangjinhao/PyGeNNProject/HMAM/output -name "*.png" -type f -delete

ARGS="--duration 3000"
ARGS="$ARGS --buffer"
ARGS="$ARGS --buffer-size 100"
# ARGS="$ARGS --SPARSE"
# ARGS="$ARGS --wEE 0 --wEI 0 --wIE 0 --wII 0"
ARGS="$ARGS --poisson"
ARGS="$ARGS --AreaIdx 30"
# ARGS="$ARGS --scaleSyn 0.01"
# ARGS="$ARGS --inSyn"

python HMAM.py $ARGS
