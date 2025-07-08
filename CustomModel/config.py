vis_content = []
model_content = {
    'V1': ["4"],
    # 'V2': ["23","4"],
}

specific_scale_syn = """
V1,V4,V1,E4,1
V1,P4,V1,E4,1
V1,S4,V1,E4,1
V1,E4,V1,E4,1
V1,P4,V1,S4,1.0
V1,V4,V1,S4,1.0
V1,E4,V1,S4,1.0
V1,V4,V1,V4,1.0
V1,S4,V1,V4,1.0
"""

stim = {
    'V1': {
        'E4': 40.,
    }
}

record_I = {
    'V1': ["E4", "P4", "V4", "S4"],
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
          'E4':  1,'S4':  1/2,'V4':  1/2,'P4':  1/2,
          'E5':  1,'S5':  1,'V5':  1,'P5':  1,
          'E6':  1,'S6':  1,'V6':  1,'P6':  1,
        },
    'beta_norm':{"H1" : 3.9,
                 "E23" : 0.71, "S23" : 1., "P23" : 0.48, "V23" :0.9,
                 "E4" : 1.66/4, "S4" : 0.24, "P4" : 0.8, "V4" : 0.46,
                 "E5" : 0.95, "S5" : 0.48, "P5" :1.09, "V5" : 1.2,
                 "E6" : 1.12, "S6" : 0.63, "P6" : 0.42, "V6" : 0.5,},

    'input':{
        "H1": 420.0 + -30.0,
        "E23": 420.0 + 70.0, "S23": 420.0 + -7.0,  "P23": 420.0 + -6.0,  "V23": 420.0 + -6.0,
        "E4": 420.0 + 0.0,  "S4": 420.0 + 0.0,   "P4": 420.0 + 0.0,   "V4": 420.0 + 0.0,
        "E5": 420.0 + 50.0,  "S5": 420.0 + 10.0,   "P5": 420.0 + -10.0,  "V5": 420.0 - 20.0,
        "E6": 420.0 + 50.0,  "S6": 420.0 + 0.0,    "P6": 420.0 + 0.0,    "V6": 420.0 - 10.0,
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
    't_ref': 2.0 # ms
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