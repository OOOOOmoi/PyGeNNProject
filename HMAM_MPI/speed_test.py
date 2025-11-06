from mpi4py import MPI
import numpy as np, time, socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

msg_size_MB = 50         # 每次消息大小（MB），可试 10/50/100
msg_size = msg_size_MB * 1024 * 1024
n_iter = 5

data = np.random.rand(msg_size // 8).astype(np.float64)  # 8 bytes each
recvbuf = np.empty_like(data)

print(f"[{rank}] host={socket.gethostname()}", flush=True)
comm.Barrier()

bw_matrix = np.zeros((size, size))

for peer in range(size):
    if peer == rank: continue
    comm.Barrier()   # 保证各 pair 在统计区有较一致的起点时刻
    t0 = time.perf_counter()

    reqs = []
    for i in range(n_iter):
        comm.Alltoall([data, MPI.DOUBLE], [recvbuf, MPI.DOUBLE])


    MPI.Request.Waitall(reqs)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    total_MB = 2 * msg_size_MB * n_iter
    bw = total_MB / elapsed if elapsed > 0 else 0.0
    bw_matrix[rank, peer] = bw
    print(f"[{rank}] peer {peer} bw {bw:.1f} MB/s (elapsed {elapsed:.3f}s)", flush=True)

comm.Barrier()
all_bw = comm.gather(bw_matrix, root=0)
if rank == 0:
    # 合并并打印矩阵
    combined = np.zeros((size,size))
    cnt = np.zeros((size,size))
    for m in all_bw:
        mask = m>0
        combined += m
        cnt += mask
    avg = combined / np.maximum(cnt, 1)
    print("\nAll-to-all avg BW (MB/s):")
    for r in range(size):
        print(" ".join(f"{avg[r,c]:7.1f}" for c in range(size)))
