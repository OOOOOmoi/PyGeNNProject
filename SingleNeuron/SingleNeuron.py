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

if __name__ == "__main__":
    modle=