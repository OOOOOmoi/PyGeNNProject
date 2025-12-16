import os
import yaml
import pickle
import pandas as pd
from pathlib import Path
current_file = Path(__file__)
parent_dir = current_file.parent.parent
data_dir = parent_dir / 'HMAM' / 'out_1mm2'
net_dir = os.path.join(data_dir, 'net.yaml')

vis_content = ['']

with open(net_dir, 'r') as f:
    net = yaml.safe_load(f)

layer_map = {
    'I': '1',
    'II/III': '23',
    'IV': '4',
    'V': '5',
    'VI': '6'
}

input={
    "E23": 200.0 + 0.0, "I23": 200.0 + -0.0,
    "E4": 200.0 + 0.0,  "I4": 200.0 + 0.0,
    "E5": 200.0 + 0.0,  "I5": 200.0 + 0.0,
    "E6": 200.0 + 0.0,  "I6": 200.0 + 0.0,
}

expLIF_dict = {
    # Leak potential of the neurons .
    'E_L': -60.0, # mV
    # Threshold potential of the neurons .
    'V_th': 20.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn_ex': 2, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0, # ms
    'DeltaT': 5.0, # mV
    'VT': -50.0, # mV
}

def has_key_path(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return False
    return True

def get_NN():
    NN_path=os.path.join(data_dir, 'neuron_numbers.pkl')
    with open(NN_path, 'rb') as f:
        NN= pickle.load(f)
    return NN

def get_SN():
    SN_path=os.path.join(data_dir, 'synapses_internal.pkl')
    with open(SN_path, 'rb') as f:
        SN= pickle.load(f)
    SN_ext_path=os.path.join(data_dir, 'synapses_external.pkl')
    with open(SN_ext_path, 'rb') as f:
        SN_ext= pickle.load(f)
    return SN, SN_ext

def get_weight():
    weight_path=os.path.join(data_dir, 'weights.pkl')
    with open(weight_path, 'rb') as f:
        weight= pickle.load(f)
    weight_sd_path=os.path.join(data_dir, 'weights_sd.pkl')
    with open(weight_sd_path, 'rb') as f:
        weight_sd= pickle.load(f)
    return weight, weight_sd

def get_weight_ext():
    weight_ext_path=os.path.join(data_dir, 'weights_ext.pkl')
    with open(weight_ext_path, 'rb') as f:
        weight_ext= pickle.load(f)
    weight_ext_sd_path=os.path.join(data_dir, 'weights_ext_sd.pkl')
    with open(weight_ext_sd_path, 'rb') as f:
        weight_ext_sd= pickle.load(f)
    return weight_ext, weight_ext_sd

def get_ext_rate():
    rate_ext_path=os.path.join(data_dir, 'rate_ext.pkl')
    with open(rate_ext_path, 'rb') as f:
        rate_ext=pickle.load(f)
    return rate_ext

def externalRates(param, eta_ext, K, W):
    # neuron parameters
    tau_m_E = param['tau_m']
    tau_syn_E = param['tau_syn_ex']
    C_m_E = param['C_m']
    V_th_E = param['V_th']
    E_L_E = param['E_L']
    # conversion factors 1/ms -> mV
    conversion_E = tau_m_E * K * tau_syn_E * W / C_m_E
    rates = 1e3 * (V_th_E - E_L_E) * eta_ext / conversion_E
    return rates

def get_cc_delay():
    delay_path=os.path.join(data_dir, 'delay_cc.pkl')
    with open(delay_path, 'rb') as f:
        delay= pickle.load(f)
    delay_sd_path=os.path.join(data_dir, 'delay_cc_sd.pkl')
    with open(delay_sd_path, 'rb') as f:
        delay_sd= pickle.load(f)
    return delay, delay_sd

def getModelName(args):
    model_name = f"{args.duration/1000.0:.1f}s"
    if args.buffer:
        model_name += f"_buffer{args.buffer_size/1000:.1f}s"
    if args.SPARSE:
        model_name += f"_SPARSE"
    if ("free_scale_input" in args):
        model_name += f"_free{args.free_scale_input}"
    if ("free_scale_syn" in args):
        model_name += f"_free{args.free_scale_syn}"
    if ("scale_stim" in args):
        model_name += f"_free{args.scale_stim}"
    if args.wEE is not None:
        model_name += f"_wEE{args.wEE}"
    if args.wEI is not None:
        model_name += f"_wEI{args.wEI}"
    if args.wIE is not None:
        model_name += f"_wIE{args.wIE}"
    if args.wII is not None:
        model_name += f"_wII{args.wII}"
    return model_name

def remove_dash_from_index_columns(df):
    # 处理 index
    if hasattr(df, "index"):
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.map(lambda x: tuple(s.replace("-", "") for s in x))
        else:
            df.index = df.index.str.replace("-", "")
    
    # 处理 columns
    if hasattr(df, "columns"):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.map(lambda x: tuple(s.replace("-", "") for s in x))
        else:
            df.columns = df.columns.str.replace("-", "")
    
    return df
