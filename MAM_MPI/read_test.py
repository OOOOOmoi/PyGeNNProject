from CustomModel_MPI import prepare
import numpy as np
from scipy.io import savemat

NN, SN, weight, _, delayMap, area_list, pop_list = prepare()

# 选一个脑区
area = sorted(SN.keys())[25]
pop_idx = {pop: i for i, pop in enumerate(pop_list)}
conn_matrix = np.zeros((len(pop_list), len(pop_list)))
for tar_pop, src_areas_dict in SN[area].items():
    i = pop_idx[tar_pop]

    for src_area, src_pops in src_areas_dict.items():
        # 只保留脑区内连接
        if src_area != area:
            continue

        for src_pop, value in src_pops.items():
            neu_num = NN[area][tar_pop]
            j = pop_idx[src_pop]
            conn_matrix[i, j] += value/neu_num
savemat(
    f"intra_area_pop_connectivity_{area}.mat",
    {
        "conn_matrix": conn_matrix,
        "area": area,
        "pop_list": pop_list
    }
)
