from mpi4py import MPI



def estimate_offset_master_peer(comm, peer, niter=10):
    offsets = []
    rtts = []
    for i in range(niter):
        t0 = MPI.Wtime()
        comm.send(None, dest=peer, tag=3000+i)   # ping
        t1 = comm.recv(source=peer, tag=3000+i)  # t1 is worker's recv-time (worker sends it)
        t2 = MPI.Wtime()
        rtt = t2 - t0
        offset = (t0 + t2) / 2.0 - t1
        offsets.append(offset)
        rtts.append(rtt)
    # choose offset corresponding to minimal RTT samples to reduce asymmetry effect
    k = max(1, niter//4)
    idx = sorted(range(len(rtts)), key=lambda i: rtts[i])[:k]
    avg_offset = sum(offsets[i] for i in idx) / len(idx)
    return avg_offset

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()   # 包含 master
num_workers = max(0, size - 1)

if rank == 0:
    print("[MASTER] estimating clock offsets ...", flush=True)
    offsets = {}
    for peer in range(1, size):
        offsets[peer] = estimate_offset_master_peer(comm, peer)
        print(f"  offset to rank {peer}: {offsets[peer]:+.6f} sec", flush=True)
    print("[MASTER] clock offset estimation done.", flush=True)