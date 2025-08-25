import pandas as pd
from config import get_weight
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'out')

# weight and syn proccess
# weight_path=os.path.join(data_dir, 'weights.pkl')
# weight_sd_path=os.path.join(data_dir, 'weights_sd.pkl')
# syn_num_path=os.path.join(data_dir, 'synapses_internal.pkl')
# df = pd.read_pickle(weight_path)
# df_sd = pd.read_pickle(weight_sd_path)
# df_syn = pd.read_pickle(syn_num_path)

# with open("connectivity.txt", "w") as f:
#     # 写表头
#     f.write("tar_area,tar_layer,tar_pop,src_area,src_layer,src_pop,syn_num,w,w_sd\n")
    
#     # 遍历索引 (tar)
#     for (tar_area, tar_layer, tar_pop), row in df.iterrows():
#         tar = (tar_area, tar_layer, tar_pop)
#         # 遍历列 (src)
#         for (src_area, src_layer, src_pop), value in row.items():
#             src = (src_area, src_layer, src_pop)
#             w_sd = df_sd.loc[tar, src]
#             syn_num = df_syn.loc[tar, src]
#             f.write(f"{tar_area},{tar_layer},{tar_pop},{src_area},{src_layer},{src_pop},{syn_num},{value},{w_sd}\n")

# neuron number
NN_path=os.path.join(data_dir, 'neuron_numbers.pkl')
with open(NN_path, 'rb') as f:
    NN = pd.read_pickle(f)
with open("neuron_numbers.txt", "w") as f:
    f.write("area,layer,pop,neuron_num\n")
    for (area, layer, pop), num in NN.items():
        f.write(f"{area},{layer},{pop},{num}\n")
