from mpi4py import MPI
import socket, os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()
pid = os.getpid()

print(f"[Rank {rank}/{size}] 启动成功 on host={host}, pid={pid}", flush=True)

# ---- 全体同步 ----
comm.Barrier()
if rank == 0:
    print("\n=== 所有进程已启动，开始通信测试 ===\n", flush=True)

# ---- 广播测试 ----
data = {"msg": "hello from rank 0", "time": MPI.Wtime()} if rank == 0 else None
data = comm.bcast(data, root=0)

comm.Barrier()
print(f"[Rank {rank}] 收到广播: {data}", flush=True)

# ---- Gather 测试 ----
info = {"rank": rank, "host": host, "pid": pid}
all_info = comm.gather(info, root=0)

comm.Barrier()
if rank == 0:
    print("\n=== Gather 收到的所有进程信息 ===", flush=True)
    for i in all_info:
        print(i, flush=True)
    print("\n=== 调试测试完成 ===", flush=True)
