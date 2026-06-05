#!/bin/bash

BASE_DIR="data_4.5T"
MAX_JOBS=4   # 并发数（自己调）

compress() {
    dir="$1"
    archive="${dir}.tar.zst"
    lock="${dir}.lock"

    # 已压缩或正在处理就跳过
    [ -f "$archive" ] && return
    [ -f "$lock" ] && return

    touch "$lock"

    echo "Compressing $dir ..."

    tar -I 'zstd -T2' -cf "$archive" "$dir"

    if [ $? -eq 0 ]; then
        rm -rf "$dir"
        rm -f "$lock"
        echo "Done $dir"
    else
        rm -f "$lock"
        echo "Failed $dir"
    fi
}

export -f compress

while true; do
    for dir in "$BASE_DIR"/*ms/; do
        [ -d "$dir" ] || continue
        
        dirname="${dir%/}"

        # 控制并发
        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
            sleep 1
        done

        compress "$dirname" &
    done

    sleep 3
done