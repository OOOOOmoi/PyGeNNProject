TARGET_DIR="./HMAM_MPI_CODE"
if [ -d "$TARGET_DIR" ]; then
    echo "清空目录 $TARGET_DIR 中的所有内容..."
    rm -rf "$TARGET_DIR"/*
else
    echo "目录 $TARGET_DIR 不存在，正在创建..."
    mkdir -p "$TARGET_DIR"
fi

mpirun -np 35 \
    -host 172.22.163.209:18,172.22.163.210:17 \
    --mca btl_tcp_if_include bond0 \
    --mca oob_tcp_if_include bond0 \
    -x CUDA_PATH \
    --output-filename logs/out \
    /home/yangjinhao/miniconda3/envs/pygenn52/bin/python \
    HMAM_MPI_MM.py > output.txt