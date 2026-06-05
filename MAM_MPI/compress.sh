#!/bin/bash

start=990
end=990      # 仿真总时间
step=10

for ((t=$start; t<=$end; t+=$step)); do
    next=$(echo "$t + $step" | bc)
    dir="${t}.0-${next}.0ms"
    archive="compress/${dir}.tar.zst"
    
    # 等待目录出现
    while [ ! -d "data_4.5T/$dir" ]; do
        sleep 5
    done
    
    echo "Compressing $dir ..."
    
    tar -I 'zstd -T0' -cf "data_4.5T/$archive" "data_4.5T/$dir"
    
    if [ $? -eq 0 ]; then
        rm -rf "data_4.5T/$dir"
    fi
done