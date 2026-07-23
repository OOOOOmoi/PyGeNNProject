#!/bin/bash
# Wang 2002 Decision-Making Model — PyGeNN Runner
# Usage: bash runner.sh [coherence] [seed]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygenn52
export CUDA_PATH=/home/yangjinhao/CUDA/cuda-12.0
export CUDA_HOME=/home/yangjinhao/CUDA/cuda-12.0

COH=${1:-51.2}
SEED=${2:-4}

cd "$(dirname "$0")"

python3 -c "
from wang2002_pygenn import Wang2002Sim, modelparams
stimparams = {'Ton': 500.0, 'Toff': 1500.0, 'mu0': 0.040, 'coh': ${COH}}
sim = Wang2002Sim(modelparams, stimparams, dt_ms=0.02)
sim.build(model_name='wang2002_coh${COH}_seed${SEED}', rng_seed=${SEED})
sim.run(2000.0, rng_seed=${SEED}, report_interval=200.0, sum_update_steps=20)
sim.save_spikes('spikesE_coh${COH}.txt', 'spikesI_coh${COH}.txt')
sim.plot_raster('wang2002_raster_coh${COH}.pdf')
sim.plot_firing_rates('wang2002_rates_coh${COH}.pdf')
"
