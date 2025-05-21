import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime
import random
import string

popName='E23'
filename = f"/home/yangjinhao/PyGenn/SingleColumn/output/spike/{popName}_spikes.csv"
df = pd.read_csv(filename, comment='#', names=["Time", "NeuronID"])
df = df[df["Time"] >= 100]  # 省略前100ms数据
plt.figure(figsize=(10, 6))
plt.scatter(df["Time"], df["NeuronID"], s=2, c='blue')
plt.savefig(f"{popName}_raster.png")
