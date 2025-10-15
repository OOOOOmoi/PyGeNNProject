mpirun -np 35 \
    -host 172.22.163.209:18,172.22.163.210:17 \
    --mca btl_tcp_if_include bond0 \
    --mca oob_tcp_if_include bond0 \
    -x CUDA_PATH \
    /home/yangjinhao/miniconda3/envs/pygenn52/bin/python \
    HMAM_MPI_MM.py 