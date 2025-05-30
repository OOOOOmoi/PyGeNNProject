import numpy as np
from scipy.stats import norm
from argparse import ArgumentParser
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from scipy.stats import norm
from time import perf_counter
from itertools import product
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict,defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
DataPath=os.path.join(current_dir, "custom_Data_Model_3396.json")
with open(DataPath,'r') as f:
    ParamOfAll=json.load(f)
SynapsesWeightMean=OrderedDict()
SynapsesWeightSd=OrderedDict()
SynapsesNumber=OrderedDict()
NeuronNumber=OrderedDict()
Dist=OrderedDict()