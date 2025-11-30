# find /home/yangjinhao/PyGeNNProject/HMAM/output -name "*.png" -type f -delete

ARGS="--duration 1000"
ARGS="$ARGS --buffer"
ARGS="$ARGS --buffer-size 100"
# ARGS="$ARGS --SPARSE"
# ARGS="$ARGS --wEE 10 --wEI 40 --wIE 60 --wII 50"
# ARGS="$ARGS --specificW"
ARGS="$ARGS --poisson"
# ARGS="$ARGS --AreaIdx 0"
ARGS="$ARGS --device 1"
# ARGS="$ARGS --scaleNeu 150"
# ARGS="$ARGS --scaleSyn 0.0000001"
# ARGS="$ARGS --inSyn"

python HMAM.py $ARGS
