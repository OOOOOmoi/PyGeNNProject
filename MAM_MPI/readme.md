# MAM_MPI — Macaque Visual Cortex Large-Scale Spiking Neural Network Simulation

A large-scale Spiking Neural Network (SNN) simulation platform based on **PyGeNN + Numba CUDA + multiprocessing parallelism**, designed for simulating neuronal population activity in the macaque visual cortex.

---

## Technical Architecture

| Layer | Technology Stack |
|-------|------------------|
| Neuron Model | LIF (Leaky Integrate-and-Fire) |
| Synapse Model | ExpCurr (exponential decay current synapse) |
| GPU Acceleration | NVIDIA CUDA (Numba custom kernels) |
| Parallel Framework | Python `multiprocessing` (multi-GPU, multi-process) |
| Network Modeling | [PyGeNN](https://github.com/genn-team/genn) |
| Visualization | matplotlib (raster, hist, PSD, firing rate curves) |

## Model Scale

- **32 visual cortical areas**: V1, V2, V3, V4, MT, MSTd, LIP, FEF, 7a, 46, AITd, etc.
- **Each area contains 5 cortical layers** (Layer 1, 2/3, 4, 5, 6)
- **Each layer contains 4–5 neuron types**: E (excitatory), S, P, V, H (inhibitory), totaling ~17 subtypes
- Each neuron type has independent biophysical parameters (membrane capacitance, leak conductance, refractory period, resting potential, threshold, etc.)

## Dependencies

- Python 3.8+
- [PyGeNN](https://github.com/genn-team/genn)
- Numba (CUDA JIT compilation)
- NumPy, SciPy, Pandas, Matplotlib
- nested_dict

## Quick Start

```bash
# Basic run (default parameters)
python CustomModel_MPI.py --duration 1000

# Full run (32 areas + scaling factor + synaptic current recording)
python CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0 --inSyn

# Run with stimulation
python CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0 --stim --stim-start 300 --stim-end 800

# Use model with different cortical surface area
python CustomModel_MPI.py --duration 1000 --AreaNum 32 --surface 3396
```

Or run via script:

```bash
bash runner.sh
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--duration` | float | 500.0 | Simulation duration (ms) |
| `--AreaNum` | int | — | Number of brain areas to simulate (1–32) |
| `--scale` | float | 1.0 | Inter-area connection weight scaling factor |
| `--surface` | float | — | Cortical surface area (mm²), affects which model data file is loaded |
| `--stim` | flag | False | Whether to apply external stimulation |
| `--stim-start` | float | 300 | Stimulation start time (ms) |
| `--stim-end` | float | 800 | Stimulation end time (ms) |
| `--inSyn` | flag | False | Whether to record synaptic currents and membrane potentials |
| `--buffer` | flag | False | Whether to use a buffer for spike storage |
| `--buffer-size` | int | 100 | Recording buffer size |

## Project Structure

```
MAM_MPI/
├── CustomModel_MPI.py      # Main program: model construction, simulation loop, multi-process coordination, CUDA kernels
├── config.py               # Model parameter configuration (neurons, synapses, connections, stimulation)
├── getStruct.py            # Model structure construction (weight matrix, delay matrix, network topology)
├── visual.py               # Visualization: raster plots, firing rates, PSD power spectra
├── record.py               # Data recording: spikes, synaptic currents, membrane potentials
├── psd.py                  # Power spectral density analysis
├── rate_curve.py           # Firing rate curve analysis
├── connectom.py            # Connectome matrix plotting
├── inSyn.py                # Synaptic current recording functions
├── update_time.py          # Communication time visualization
├── read_test.py            # Test/debug script
├── runner.sh               # Run script
├── runner_multi.sh         # Multi-task run script
├── GenCODE/                # PyGeNN-generated CUDA code
├── log/                    # Detailed runtime records
└── output/                 # Visualization output (raster, hist, psd, rate)
    └── spike/              # Spike data storage (CSV)
```

## Key Technical Highlights

### 1. Custom Numba CUDA Kernels

`fast_update_inSyn_gpu`: Efficiently computes Poisson noise synaptic input on GPU using the XOROSHIRO128+ random number generator:
- Small-scale trials (n_trials < 2048): exact binomial distribution sampling + shared memory reduction
- Large-scale trials: Poisson approximation for acceleration

### 2. Spike Buffer Ring Buffer

Handles cross-area axonal conduction delays — non-local area connections reach target neurons after delay steps via the spike buffer.

### 3. Multi-GPU Multi-Process Parallelism

- Uses 4 GPUs, each GPU can run multiple processes
- The main process collects spike counts from each subprocess via `multiprocessing.Queue`, aggregates and broadcasts them
- After simulation ends, complete spike data is collected for visualization

### 4. Model Data Files

Model connectivity structure (synapse counts, neuron counts, distance matrices) is loaded from JSON parameter files (`model_info_schmidt_motif_diff_s/`), supporting model variants with different surface areas (e.g., 3396 mm²).

## Data Directories

- **Generated Code**: `GenCODE/` — GPU code auto-generated by PyGeNN
- **Runtime Logs**: `log/` — Time statistics for each worker (stepping, communication, inSyn updates, etc.)
- **Visualization Results**: `output/` — raster plots, histograms, PSD, firing rate curves, etc.

## References

Based on the macaque visual cortex microcircuit motif model by Schmidt et al., extended across multiple cortical areas into a large-scale whole-brain simulation.
