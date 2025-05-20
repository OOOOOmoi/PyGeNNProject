import os
import json
from collections import OrderedDict
DataPath=os.path.join("custom_Data_Model_3396.json")
file1=os.path.join("SynapsesWeight.txt")
file3=os.path.join("SynapsesNumber.txt")
file2=os.path.join("NeuronNumber.txt")
with open(DataPath,'r') as f:
    ParamOfAll=json.load(f)
SynapsesWeightMean=OrderedDict()
SynapsesWeightSd=OrderedDict()
NeuronNumber=OrderedDict()
Dist=OrderedDict()
AreaList=['V1','V2']
PopList=ParamOfAll['population_list']
SynapsesWeightMean=ParamOfAll["synapse_weights_mean"]
SynapsesWeightSd=ParamOfAll["synapse_weights_sd"]
SynapsesNumber=ParamOfAll["synapses"]
NeuronNumber=ParamOfAll["neuron_numbers"]
Dist=ParamOfAll["distances"]
# 打开两个文件，分别保存数据
with open(file1, 'w') as f1, open(file3, 'w') as f3:
    # 用于存储 popTar 和 wEx 的集合，避免重复写入
    popTar_wEx_written = set()
    for areaTar in AreaList:
        for popTar in PopList:
            for areaSrc in AreaList:
                for popSrc in PopList:
                    # 从数据字典中获取 wAve, wSd, wEx
                    wAve = SynapsesWeightMean[areaTar][popTar][areaSrc][popSrc]
                    wSd = SynapsesWeightSd[areaTar][popTar][areaSrc][popSrc]
                    wAveEx = SynapsesWeightMean[areaTar][popTar]["external"]["external"]
                    wSdEx = 0
                    synNum = SynapsesNumber[areaTar][popTar][areaSrc][popSrc]
                    synNumEx = SynapsesNumber[areaTar][popTar]['external']['external']
                    # if areaTar+popTar not in popTar_wEx_written:
                    #     f1.write(f"{areaTar} {popTar} external external {wAveEx} {wSdEx}\n")
                    #     f3.write(f"{areaTar} {popTar} external external {synNumEx}\n")
                    #     popTar_wEx_written.add(areaTar+popTar)  # 记录 popTar，防止重复写入

                    # 写入第一个文件：popSrc popTar wAve wSd
                    # if synNum!=0:
                    f1.write(f"{areaTar} {popTar} {areaSrc} {popSrc} {wAve} {wSd}\n")
                    f3.write(f"{areaTar} {popTar} {areaSrc} {popSrc} {synNum}\n")
                    # 如果 popTar 尚未写入第二个文件，写入 popTar 和 wEx

with open(file2, 'w') as f2:
    for areaTar in AreaList:
        for popTar in PopList:
            neuronNum=NeuronNumber[areaTar][popTar]
            f2.write(f"{areaTar} {popTar} {neuronNum}\n")
                    
with open(file4, 'w') as f4:
    for areaTar in AreaList:
        for areaSrc in AreaList:
            d=Dist[areaTar][areaSrc]
            f4.write(f"{areaTar} {areaSrc} {d}\n")
