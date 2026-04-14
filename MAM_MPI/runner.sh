#!/bin/bash
find /home/yangjinhao/PyGeNNProject/MAM_MPI/output -name "*.csv" -type f -delete
duration=300
AreaNum=32
stim_start=100
stim_end=200
python CustomModel_MPI.py \
    --duration $duration \
    --AreaNum $AreaNum \
    --stim-start $stim_start \
    --stim-end $stim_end \
    --scale 1 \
    # --inSyn