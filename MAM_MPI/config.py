vis_content = ['area-psd']
model_content = {
    "V1": [],
    "V1": [],
    "V2": [],
    "VP": [],
    "V3": [],
    "V3A": [],
    "MT": [],
    "V4t": [],
    "V4": [],
    "VOT": [],
    "MSTd": [],
    "PIP": [],
    "PO": [],
    "DP": [],
    "MIP": [],
    "MDP": [],
    "VIP": [],
    "LIP": [],
    "PITv": [],
    "PITd": [],
    "MSTl": [],
    "CITv": [],
    "CITd": [],
    "FEF": [],
    "TF": [],
    "AITv": [],
    "FST": [],
    "7a": [],
    "STPp": [],
    "STPa": [],
    "46": [],
    "AITd": [],
    "TH": [],
    # 'V2': ["23","4"]
}

specific_scale_syn = """
"""

stim = {
    'V1': {
        'E23': 500.,
    }
}

record_I = {
    'V1': ["E4", "S4", "P4", "V4"],
}

connection_params = {
    # Relative inhibitory synaptic strength (in relative units).
    'g': -16.,
    'g_H': -2.,
    'g_V': -2.,
    'g_P': -2.,
    'g_S': -2.,
    
    'alpha_norm': {
          'H1':  1,
          'E23': 1,'S23': 1,'V23': 1,'P23': 1,  
          'E4':  1,'S4':  1,'V4':  1,'P4':  1,
          'E5':  1,'S5':  1,'V5':  1,'P5':  1,
          'E6':  1,'S6':  1,'V6':  1,'P6':  1,
        },
    'beta_norm':{"H1" : 3.9,
                 "E23" : 0.8, "S23" : 0.7, "P23" : 0.5, "V23" :1,
                 "E4" : 1.66/4, "S4" : 0.24, "P4" : 0.8, "V4" : 0.46,
                 "E5" : 0.95, "S5" : 0.48, "P5" :1.09, "V5" : 1.2,
                 "E6" : 1.12, "S6" : 0.63, "P6" : 0.42, "V6" : 0.5,},

    'input':{
        "H1": 420.0 + -0.0,
        "E23": 420.0 + 0.0, "S23": 420.0 + -0.0,  "P23": 420.0 + -0.0,  "V23": 420.0 + -0.0,
        "E4": 420.0 + 0.0,  "S4": 420.0 + 0.0,   "P4": 420.0 + 0.0,   "V4": 420.0 + -0.0,
        "E5": 420.0 + 0.0,  "S5": 420.0 + 0.0,   "P5": 420.0 + -0.0,  "V5": 420.0 - 0.0,
        "E6": 420.0 + 0.0,  "S6": 420.0 + 0.0,    "P6": 420.0 + 0.0,    "V6": 420.0 - 0.0,
    },

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight 
    'PSP_e': 0.15, # mV
    'PSP_e_23_4': 0.3, #mV
    'PSP_e_5_h1': 0.15, #mV
    # synaptic weight  for external input
    'PSP_ext': 0.15, #mV
    
    # relative SD of normally distributed synaptic weights
    'PSC_rel_sd_normal': 0.1,
    # relative SD of lognormally distributed synaptic weights
    'PSC_rel_sd_lognormal': 3.0,

    # scaling factor for cortico-cortical connections (chi)
    'cc_weights_factor': 1.,
    # factor to scale cortico-cortical inh. weights in relation
    # to exc. weights (chi_I)
    'cc_weights_I_factor': 0.8,

    'rate_ext': 10.,
}

single_neuron_dict = {
    'E_L': -70.0, # mV
    'V_th': -50.0, # mV
    'V_reset': -60.0, # mV
    'C_m': 1000.0, # pF
    'tau_m': 40.0, # ms
    'tau_syn': 0.5, # ms
    't_ref': 2.0, # ms

    'rate_ext':{
        "H1": 650,
        "E23": 930, "S23": 870.0,  "P23": 1460.0,  "V23": 1405.0,
        "E4": 890,  "S4": 2105.0,   "P4": 1984.0,   "V4": 240.0,
        "E5": 4740,  "S5": 530.0,   "P5": 930.0,  "V5": 870.0,
        "E6": 1770,  "S6": 885.0,    "P6": 1170.0,    "V6": 1620.0,
    },

    'Cm': {# pF
        "H1": 37.11,
        "E23": 123.41, "P23": 70.95, "S23": 82.34, "V23": 41.23,
        "E4": 80.16, "P4": 81.21, "S4": 132.86, "V4": 40.3,
        "E5": 149.43, "P5": 70.9, "S5": 52.32, "V5": 59.29,
        "E6": 99.96, "P6": 49.65, "S6": 96.09, "V6": 65.87,
    },
    'gL': {# nS
        "H1": 4.07,
        "E23": 2.47, "P23": 9.49, "S23": 3.17, "V23": 6.4,
        "E4": 5.16, "P4": 9.19, "S4": 7.96, "V4": 1.87,
        "E5": 16.66, "P5": 5.21, "S5": 3.43, "V5": 6.52,
        "E6": 5.88, "P6": 6.86, "S6": 2.99, "V6": 6.09,
    },
    'tref': {# ms
        "H1": 3.5,
        "E23": 3.0, "P23": 1.26, "S23": 1.85, "V23": 2.75,
        "E4": 4.4, "P4": 1.5, "S4": 2.2, "V4": 2.4,
        "E5": 4.25, "P5": 1.85, "S5": 1.9, "V5": 2.55,
        "E6": 3.3, "P6": 1.65, "S6": 2.1, "V6": 2.85,
    },
    'Vrest': {# mV
        "H1": -65.5,
        "E23": -80.97, "P23": -82.35, "S23": -69.16, "V23": -67.94,
        "E4": -72.53, "P4": -70.45, "S4": -74.2, "V4": -63.14,
        "E5": -68.28, "P5": -77.5, "S5": -70.01, "V5": -72.00,
        "E6": -77.5, "P6": -76.42, "S6": -62.99, "V6": -78.85,
    },
    'Vth': {# mV
        "H1": -40.20,
        "E23": -40.53, "P23": -56.32, "S23": -39.95, "V23": -41.34,
        "E4": -47.63, "P4": -44.23, "S4": -44.07, "V4": -40.89,
        "E5": -40.55, "P5": -51.2, "S5": -47.38, "V5": -51.2,
        "E6": -42.31, "P6": -49.06, "S6": -37.19, "V6": -44.81,
    },
}

expLIF_dict = {
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': 20.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0, # ms
    'DeltaT': 5.0, # mV
    'VT': -50.0, # mV
}

# dictionary defining single-cell parameters
single_neuron_dict_of_weight = {
    "E":{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0 # ms
    },
    "S":{
    # Leak potential of the neurons .
    'E_L': -76.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 800.0, # pF
    # Membrane time constant .
    'tau_m': 50.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    },
    "P":{
    # Leak potential of the neurons .
    'E_L': -86.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 200.0, # pF
    # Membrane time constant .
    'tau_m': 10.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    },
    'V':{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -65.0, # mV
    # Membrane capacitance .
    'C_m': 100.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    },
    'H':{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -65.0, # mV
    # Membrane capacitance .
    'C_m': 100.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    }
}



collection_params = {
    'model_content': model_content,
    'connection_params': connection_params,
    'single_neuron_dict': single_neuron_dict,
    'single_neuron_dict_of_weight': single_neuron_dict_of_weight,
    'specific_scale_syn': specific_scale_syn,
    'stim': stim,
    'type_list': ["E", "S", "P", "V", "H"],
    'expLIF_dict': expLIF_dict,
}