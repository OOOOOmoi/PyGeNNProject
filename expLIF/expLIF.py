import numpy as np
import pygenn
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var)
import matplotlib.pyplot as plt

expLIF_model = pygenn.create_neuron_model(
    "expLIF",
    params=["Vthresh", "TauM", "TauRefrac", "C", "Vrest", "Vreset", "Ioffset", "DeltaT", "VT"],
    vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE),
          ("RefracTime", "scalar", pygenn.VarAccess.READ_WRITE)],
    sim_code=
        """
        if (RefracTime <= 0.0) {
            scalar dV = (-(V - Vrest) + DeltaT * exp((V - VT) / DeltaT) + Rmembrane * (Ioffset + Isyn)) / TauM;
            V += dV * dt;
        }else {
            RefracTime -= dt;
        }
        """,
    threshold_condition_code="(RefracTime <= 0.0) && (V >= Vthresh)",
    reset_code=
        """
        V = Vreset;
        RefracTime = TauRefrac;
        """,
    derived_params=[("Rmembrane", lambda pars,dt:pars["TauM"]/pars["C"])],
)

single_neuron_dict = {
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
    'input': 380.0, # pA
    'DeltaT': 5.0, # mV
    'VT': -50.0, # mV
}

if __name__ == "__main__":
    duration = 1000.0  # ms
    model=GeNNModel("float", "expLIF")
    model.dt = 0.1  # ms
    model.timing_enabled = True
    model.default_var_location = VarLocation.HOST_DEVICE
    model.default_sparse_connectivity_location = VarLocation.HOST_DEVICE
    explif_init = {"V": init_var("Normal", {"mean": -150.0, "sd": 50.0}), "RefracTime": single_neuron_dict['t_ref']}
    explif_params = {
        "C": single_neuron_dict['C_m'] / 1000,  # Convert pF to nF
        "TauM": single_neuron_dict['tau_m'],
        "Vrest": single_neuron_dict['E_L'],
        "Vreset": single_neuron_dict['V_reset'],
        "Vthresh": single_neuron_dict['V_th'],
        "Ioffset": single_neuron_dict['input'] / 1000,
        "TauRefrac": single_neuron_dict['t_ref'],
        "DeltaT": single_neuron_dict['DeltaT'],
        "VT": single_neuron_dict['VT']
    }
    pop = model.add_neuron_population('expLIF_pop', 1, expLIF_model, explif_params, explif_init)
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