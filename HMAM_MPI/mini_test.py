from mpi4py import MPI
import time, pickle
from time import perf_counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("MASTER start", flush=True)
else:
    print(f"WORKER {rank} start", flush=True)

# 模拟计算/不同机器耗时
time.sleep(1 + rank * 0.1)

for i in range(5):
    msg = {"rank": rank, "i": i, "timestamp": perf_counter()}
    comm.gather(msg, root=0)
    ctrl = comm.bcast({"type":"continue"}, root=0)
    time.sleep(0.1)

if rank == 0:
    print("MASTER done", flush=True)
else:
    print(f"WORKER {rank} done", flush=True)
