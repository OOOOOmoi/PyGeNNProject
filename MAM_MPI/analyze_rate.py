#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 MAM_MPI V1 单区 spike CSV 的群体放电率，并与 Schmidt 实验值对比。"""
import os, glob, sys, json
import numpy as np

def analyze(spike_dir, dur_s, data_json):
    d = json.load(open(data_json))
    nn = d['neuron_numbers']['V1']
    pops = d['population_list']
    Ntot = sum(nn.values())
    rates = {}
    total_spikes = 0
    for p in pops:
        f = os.path.join(spike_dir, f'V1_{p}_spikes.csv')
        a = np.loadtxt(f, delimiter=',')
        if a.size == 0:
            a = np.empty((0, 2))
        if a.ndim == 1:
            a = a.reshape(1, -1)
        n_spikes = a.shape[0]
        total_spikes += n_spikes
        r = n_spikes / nn[p] / dur_s
        rates[p] = r
    wmean = sum(nn[p] * rates[p] for p in pops) / Ntot
    return rates, wmean, total_spikes, Ntot, pops

def main():
    spike_dir = sys.argv[1]
    dur_s = float(sys.argv[2])
    data_json = sys.argv[3] if len(sys.argv) > 3 else \
        '../model_info_schmidt_motif_diff_s/custom_Data_Model_3396.json'
    rates, wmean, total_spikes, Ntot, pops = analyze(spike_dir, dur_s, data_json)
    print(f"pop      rate(Hz)   N_total")
    for p in pops:
        print(f"{p:5s}  {rates[p]:8.2f}  {int(json.load(open(data_json))['neuron_numbers']['V1'][p]):8d}")
    vals = np.array([rates[p] for p in pops])
    print("-" * 34)
    print(f"total_spikes = {total_spikes}")
    print(f"加权平均放电率 = {wmean:.3f} Hz")
    print(f"简单平均 = {vals.mean():.3f} Hz, 中位 = {np.median(vals):.3f} Hz")
    print(f"参考实验: mean=4.72, median=3.23 Hz (pvc5, 49 units)")

if __name__ == '__main__':
    main()
