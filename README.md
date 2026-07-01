# PyGeNNProject — Macaque Visual Cortex Large-Scale Spiking Neural Network Simulation Platform

A large-scale Spiking Neural Network (SNN) simulation platform based on **PyGeNN** (GPU-enhanced Neural Networks) + **Numba CUDA** + **multiprocessing/MPI parallelism**, designed for simulating neuronal population activity in the macaque visual cortex.

---

## Overview

This project constructs large-scale SNN models ranging from single neurons and single cortical columns to multi-area full visual cortex, covering 32 visual cortical areas with multi-GPU parallel acceleration. The models are based on the macaque visual cortex microcircuit framework by Schmidt et al. and implemented via PyGeNN for efficient GPU simulation.

## Model Directory

| Directory | Description |
|-----------|-------------|
| `SingleNeuron/` | Single-neuron LIF model tests |
| `expLIF/` | Exponential LIF neuron model definition (with adaptive exponential) |
| `DualEXP/` | Dual Exponential Synapse model |
| `IzhkNeuron/` | Izhikevich neuron model implementation |
| `EIBalance/` | Excitatory-inhibitory balanced network model |
| `SingleColumn/` | Single cortical column model (Potjans-Diesmann style, 16 neuron populations) |
| `DoubleColumn/` | Dual cortical column interconnection model |
| `MultiColumn/` | Multi-column extension model |
| `MultiLayer/` | Multi-layer network model |
| `potjansModel/` | Potjans-Diesmann 2014 cortical microcircuit reproduction |
| `SchmidtModel/` | Schmidt macaque visual cortex microcircuit model |
| `CustomModel/` | Custom full-cortex connectivity model (supports 32 brain areas) |
| `HMAM/` | Human Multi-Area Model |
| `HMAM_MPI/` | MPI distributed multi-node version of HMAM |
| `MAM_MPI/` | **Main model**: Multi-Area MPI parallel model (32 areas, multi-GPU multi-process) |
| `ScalingTest/` | Model scalability tests |
| `ProjectOfStimExcNeuronInFirstArea/` | Stimulation experiment on excitatory neurons in the first area |
| `model_info_schmidt_motif_diff_s/` | Connectivity data files for Schmidt motif model |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Neuron Models | LIF / Exponential LIF / Izhikevich |
| Synapse Models | ExpCurr (exponential decay current) / Dual Exponential |
| GPU Acceleration | PyGeNN + Numba CUDA (custom kernels) |
| Parallel Frameworks | Python `multiprocessing` (multi-GPU) + MPI (multi-node) |
| Data Processing | NumPy, SciPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Build System | setuptools |

## Main Model: MAM_MPI

MAM_MPI is the most comprehensive simulation model in this project, simulating **32 macaque visual cortical areas** (V1, V2, V3, V4, MT, MSTd, LIP, FEF, 7a, 46, AITd, etc.), with each area containing 5 cortical layers and 4–5 neuron subtypes.

```bash
# Basic run
python MAM_MPI/CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0

# Run with stimulation
python MAM_MPI/CustomModel_MPI.py --duration 1000 --AreaNum 32 --scale 1.0 --stim --stim-start 300 --stim-end 800
```

See `MAM_MPI/readme.md` for details.

## Dependencies

- Python 3.8+
- [PyGeNN](https://github.com/genn-team/genn) (GPU SNN simulation framework)
- Numba (CUDA JIT compilation)
- NumPy, SciPy, Pandas, Matplotlib
- `nested_dict`, `pynvml`

```bash
pip install pandas matplotlib scipy seaborn numba nested_dict pynvml
```

## Installation

```bash
git clone git@github.com:OOOOOmoi/PyGeNNProject.git
cd PyGeNNProject
pip install -e .
```

## Data Files

- `custom_Data_Model_3396.json` — Complete connection parameters for 32 brain areas (3396 mm² cortical surface area)
- `default_Data_Model__*.json` — Model parameters for different cortical surface areas (50/70/100 mm²)
- `indegrees_full.json` / `outdegrees_full.json` — Whole-brain in-degree and out-degree data
- `viscortex_raw_data.json` — Visual cortex raw connectivity data
- `Fac_result.json` — Facilitation factor results

## References

- Schmidt et al., "A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque visual cortical areas", *PLOS Computational Biology*, 2018.
- Potjans & Diesmann, "The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model", *Cerebral Cortex*, 2014.
- Knight & Nowotny, "GPUs outperform current HPC and neuromorphic solutions in terms of speed and energy when simulating a highly-connected cortical model", *Frontiers in Neuroscience*, 2018.
