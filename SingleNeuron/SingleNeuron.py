import numpy as np
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
import matplotlib.pyplot as plt
single_neuron_dict = {
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 1000.0, # pF
    # Membrane time constant .
    'tau_m': 40.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0, # ms

    'input': 0.55, # pA
}
if __name__ == "__main__":
    duration = 1000.0  # ms
    model=GeNNModel("float", "SingleNeuronModel")
    model.dt = 0.1  # ms
    neuronParam = single_neuron_dict
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    lif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": 2.0}
    lif_params = {"C": neuronParam['C_m']/1000, "TauM": neuronParam['tau_m'],
                    "Vrest": neuronParam['E_L'], "Vreset": neuronParam['V_reset'],
                    "Vthresh" : neuronParam['V_th'], "Ioffset": neuronParam['input'],
                    "TauRefrac": neuronParam['t_ref']}
    pop = model.add_neuron_population('single_lif', 1, "LIF", lif_params, lif_init)
    pop.spike_recording_enabled = True
    model.build()
    duration_timesteps = int(round(duration / 0.1))
    model.load(num_recording_timesteps=duration_timesteps)
    V=[]
    while model.t < duration:
        model.step_time()
        pop.vars["V"].pull_from_device()
        v=pop.vars["V"].current_values
        V.append(v[0])

    plt.figure(figsize=(8, 5))
    plt.plot(V)
    plt.savefig("single_neuron_v.png")