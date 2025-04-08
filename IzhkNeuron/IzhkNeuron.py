import numpy as np
import matplotlib.pyplot as plt

from pygenn import GeNNModel

model = GeNNModel("float", "tutorial1")
model.dt = 0.1
izk_init = {"V": -65.0,
            "U": -20.0,
            "a": [0.02,     0.1,    0.02,   0.02],
            "b": [0.2,      0.2,    0.2,    0.2],
            "c": [-65.0,    -65.0,  -50.0,  -55.0],
            "d": [8.0,      2.0,    2.0,    4.0]}

pop = model.add_neuron_population("Neurons", 4, "IzhikevichVariable", {}, izk_init)

model.add_current_source("CurrentSource", "DC", pop, {"amp": 10.0}, {})

model.build()
model.load()

voltage = pop.vars["V"]

voltages = []
while model.t < 200.0:
    model.step_time()
    voltage.pull_from_device()
    voltages.append(voltage.values)

# Stack voltages together into a 2000x4 matrix
voltages = np.vstack(voltages)

# Create figure with 4 axes
fig, axes = plt.subplots(4, sharex=True, figsize=(15, 8))

# Plot voltages of each neuron in
for i, t in enumerate(["RS", "FS", "CH", "IB"]):
    axes[i].set_title(t)
    axes[i].set_ylabel("V [mV]")
    axes[i].plot(np.arange(0.0, 200.0, 0.1), voltages[:,i])

axes[-1].set_xlabel("Time [ms]");
plt.savefig('tutorial1.jpg')