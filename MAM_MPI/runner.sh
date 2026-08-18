#!/bin/bash
find /home/yangjinhao/PyGeNNProject/MAM_MPI/output -name "*.csv" -type f -delete
ARGS="--duration 10000"
ARGS="$ARGS --AreaNum 32"
# ARGS="$ARGS --scale 5"
# ARGS="$ARGS --stim-start 100"
# ARGS="$ARGS --stim-end 200"
# ARGS="$ARGS --inSyn"
nohup python CustomModel_MPI.py $ARGS &
