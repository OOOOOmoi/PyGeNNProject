The main GeNN+MPI code file is `HMAM_MPI.py`.
Input files are from the `out_*mm2` folder; you can modify the file name in the config to simulate cortical columns of different surface areas.
No command-line input is required. You can adjust the number of `num_worker` and `num_gpu` in the `main` function to use different numbers of GPUs to simulate different numbers of brain areas.
Output files are saved in the `output/` folder.

`HMAM_MPI_MM.py` is the multi-node version. After configuring inter-node communication, run `runner_MM.sh` to execute.
Depending on the environment, you may need to change the IP address and network interface name.
