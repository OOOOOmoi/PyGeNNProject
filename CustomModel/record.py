import os
import glob
import numpy as np
def record_spike(neuron_population, spike_data):
    for area, pop_dict in neuron_population.items():
        for pop, p in pop_dict.items():
            spike_times, spike_ids = p.spike_recording_data[0]
            spike_data[area][pop].append(np.column_stack((spike_times, spike_ids)))

def save_spike(spike_data):
    for area, pop_dict in spike_data.items():
        output_dir = f"output/spike/{area}"
        os.makedirs(output_dir, exist_ok=True)

        # 清除该区域下所有旧的 .csv 文件
        for old_file in glob.glob(f"{output_dir}/*.csv"):
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Warning: Failed to delete {old_file}: {e}")

        for pop, data_chunks in pop_dict.items():
            if len(data_chunks) == 0:
                continue  # 避免没有数据时报错
            all_data = np.vstack(data_chunks)
            save_path = f"{output_dir}/{area}_{pop}_spikes.csv"
            np.savetxt(
                save_path,
                all_data,
                delimiter=",",
                fmt=("%f", "%d"),
                header="Times [ms], Neuron ID"
            )


def record_inSyn(out_post_history, record_I, synapse_populations, PopList):
    for tar_area, tar_pop_list in record_I.items():
        for tar_pop in tar_pop_list:
            for src_pop in PopList:
                if synapse_populations[tar_area][tar_pop][tar_area][src_pop] is not None:
                    syn_pop=synapse_populations[tar_area][tar_pop][tar_area][src_pop]
                    syn_pop.out_post.pull_from_device()
                    out_post_array = syn_pop.out_post.view[:,:100]
                    if isinstance(out_post_history[tar_area][tar_pop][tar_area][src_pop], dict):
                        out_post_history[tar_area][tar_pop][tar_area][src_pop] = []
                    out_post_history[tar_area][tar_pop][tar_area][src_pop].append(out_post_array.copy())

def save_inSyn(out_post_history):
    for tar_area in out_post_history:
        for tar_pop in out_post_history[tar_area]:
            for src_area in out_post_history[tar_area][tar_pop]:
                for src_pop in out_post_history[tar_area][tar_pop][src_area]:
                    data = out_post_history[tar_area][tar_pop][src_area][src_pop]

                    if not data:
                        continue  # 空数据跳过

                    all_data = np.vstack(data)  # 合并所有时间片
                    folder_path = os.path.join("output", "inSyn", tar_area, tar_pop)
                    os.makedirs(folder_path, exist_ok=True)

                    filename = f"{src_pop}_2_{tar_pop}.csv"
                    file_path = os.path.join(folder_path, filename)

                    np.savetxt(file_path, all_data, delimiter=",", fmt="%.3f")