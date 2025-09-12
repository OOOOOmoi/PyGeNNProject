import pickle
import os
import numpy as np
from config import get_NN, get_SN, net

if __name__ == "main":
    NN = get_NN()
    SN, SN_ext = get_SN()
    area_list = net["area_list"]
    layer_list = net["layer_list"]
    pop_list = net["population_list"]
    