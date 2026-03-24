#!/bin/bash
duration=3000
AreaNum=32
python CustomModel_MPI.py \
    --duration $duration \
    --AreaNum $AreaNum \
    --scale 1 \
    --surface 80