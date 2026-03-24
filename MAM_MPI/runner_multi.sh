#!/bin/bash

duration=10000
AreaNum=32

for scale in $(seq 1.4 0.1 2.0)
do
    echo "Running with scale=$scale"
    nohup python CustomModel_MPI.py \
        --duration $duration \
        --AreaNum $AreaNum \
        --scale $scale \
        --surface 1 \
        > log_scale_${scale}.out 2>&1
done