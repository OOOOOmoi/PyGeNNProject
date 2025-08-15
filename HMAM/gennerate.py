import json 
import pickle
import os
from config import data_dir
from config import get_NN, get_SN, get_weight, get_weight_ext

layer_map = {
    'I': '1',
    'II/III': '23',
    'IV': '4',
    'V': '5',
    'VI': '6'
}

NN=get_NN()
data_reset=NN.reset_index()

data_reset['layer'] = data_reset['layer'].map(layer_map)

data_reset.to_json('output.json', orient='index', indent=2)