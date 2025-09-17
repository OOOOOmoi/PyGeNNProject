# find /home/yangjinhao/PyGeNNProject/HMAM/output -name "*.png" -type f -delete

ARGS="--duration 1000"
ARGS="$ARGS --buffer"
ARGS="$ARGS --buffer-size 100"
# ARGS="$ARGS --SPARSE"
# ARGS="$ARGS --wEE 0 --wEI 0 --wIE 0 --wII 0"
# ARGS="$ARGS --specificW"
ARGS="$ARGS --poisson"
ARGS="$ARGS --AreaIdx 30"
ARGS="$ARGS --device 3"
# ARGS="$ARGS --scaleNeu 10"
# ARGS="$ARGS --scaleSyn 0.0000001"
# ARGS="$ARGS --inSyn"

python HMAM.py $ARGS
