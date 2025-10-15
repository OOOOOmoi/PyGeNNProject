from config import get_SN, get_weight, remove_dash_from_index_columns, net

def split_indices(num_areas, num_workkers):
    # 平均分配索引到 num_gpus 个子列表
    indices = list(range(1, num_areas + 1))   # 生成 1 ~ num_areas
    chunk_size = (num_areas + num_workkers - 1) // num_workkers  # 向上取整
    return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workkers) if indices[i*chunk_size:(i+1)*chunk_size]]

SN, SN_ext = get_SN()
SN = remove_dash_from_index_columns(SN)
weight, weight_sd = get_weight()
weight = remove_dash_from_index_columns(weight)

area_list = net['area_list']
area_list = [s.replace("-", "") for s in area_list]

split_idx = split_indices(68, 34)  # splits[i] assigned to worker rank=i+1
for i in range(34):
    if i==0:
        continue   # rank 0 不处理
    assigned_areas = [area_list[j-1] for j in split_idx[i]]
    weight_local = weight.loc(axis=1)[assigned_areas,:,:]