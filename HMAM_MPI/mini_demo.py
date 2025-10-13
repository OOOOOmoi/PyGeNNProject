from mpi4py import MPI
import socket, sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[Rank {rank}/{size}] host={socket.gethostname()} python={sys.executable}")
