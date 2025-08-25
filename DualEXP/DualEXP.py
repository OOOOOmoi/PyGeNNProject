import numpy as np
from argparse import ArgumentParser, Namespace
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
from pygenn.cuda_backend import DeviceSelect
from time import perf_counter
from itertools import product
import os
import json
import random
import string
import matplotlib.pyplot as plt
from collections import defaultdict
from nested_dict import nested_dict
from scipy.stats import norm

DUAL_EXP = pygenn.create_weight_update_model(
    "DUAL_EXP",
    vars = [("g","scalar",)],
)