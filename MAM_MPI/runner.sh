#!/bin/bash
find /home/yangjinhao/PyGeNNProject/MAM_MPI/output -name "*.csv" -type f -delete
ARGS="--duration 1000"
ARGS="$ARGS --AreaNum 32"
ARGS="$ARGS --scale 1"
# ARGS="$ARGS --stim-start 100"
# ARGS="$ARGS --stim-end 200"
ARGS="$ARGS --inSyn"
python CustomModel_MPI.py $ARGS
