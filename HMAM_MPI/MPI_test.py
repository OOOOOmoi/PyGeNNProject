# mpi_gpu_demo.py
from mpi4py import MPI
import time, pickle, socket, os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 假设每台机器有 8 张 GPU
NUM_GPUS = 8

def get_gpu_id(rank):
    if rank == 0:
        return None  # master 不跑仿真
    return (rank - 1) % NUM_GPUS

def worker_loop():
    hostname = socket.gethostname()
    gpu_id = get_gpu_id(rank)

    # 模拟绑定 GPU（真实跑 CUDA 时可以设置环境变量）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for step in range(3):
        # 模拟计算
        time.sleep(0.5 + rank * 0.05)

        msg = {
            "rank": rank,
            "hostname": hostname,
            "gpu_id": gpu_id,
            "step": step,
            "timestamp": time.time()
        }
        # 发送数据到 master
        comm.gather(msg, root=0)

        # 等待 master 的广播
        ctrl = comm.bcast(None, root=0)
        print(f"[Worker {rank} | GPU {gpu_id}] recv broadcast: {ctrl}")

def master_loop():
    for step in range(3):
        gathered = comm.gather(None, root=0)
        worker_msgs = [m for m in gathered if m is not None]

        print(f"\n=== Step {step} ===")
        for msg in worker_msgs:
            latency = time.time() - msg["timestamp"]
            size_kb = len(pickle.dumps(msg)) / 1024
            print(f"Recv from rank {msg['rank']} on {msg['hostname']} "
                  f"(GPU {msg['gpu_id']}): "
                  f"latency {latency*1000:.2f} ms, size {size_kb:.1f} KB")

        # 广播控制消息
        comm.bcast({"type": "continue", "step": step}, root=0)

if __name__ == "__main__":
    if rank == 0:
        master_loop()
    else:
        worker_loop()
