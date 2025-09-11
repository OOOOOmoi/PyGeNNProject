find /home/yangjinhao/PyGeNNProject/HMAM/output -name "*.png" -type f -delete

ARGS="--duration 3000"
# ARGS="$ARGS --buffer"
# ARGS="$ARGS --buffer-size 10"
# ARGS="$ARGS --SPARSE"
ARGS="$ARGS --wEE 0.5 --wEI 10 --wIE 3 --wII 8"
ARGS="$ARGS --poisson"
# ARGS="$ARGS --scaleSyn 0.01"
# ARGS="$ARGS --inSyn"

python HMAM.py $ARGS
