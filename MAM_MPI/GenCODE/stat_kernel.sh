#!/bin/bash

echo "filename line_neuron grid_neuron block_neuron line_syn grid_syn block_syn"

for d in */ ; do
    neuron_file="$d/neuronUpdate.cc"
    syn_file="$d/synapseUpdate.cc"

    line_neuron=0
    grid_neuron="-"
    block_neuron="-"

    line_syn=0
    grid_syn="-"
    block_syn="-"

    if [ -f "$neuron_file" ]; then
        line_neuron=$(wc -l < "$neuron_file")

        grid_neuron=$(grep -B5 "updateNeuronsKernel<<<" "$neuron_file" \
            | grep "const dim3 grid" \
            | tail -1 \
            | sed -E 's/.*grid\(([0-9]+).*/\1/')

        block_neuron=$(grep -B5 "updateNeuronsKernel<<<" "$neuron_file" \
            | grep "const dim3 threads" \
            | tail -1 \
            | sed -E 's/.*threads\(([0-9]+).*/\1/')
    fi

    if [ -f "$syn_file" ]; then
        line_syn=$(wc -l < "$syn_file")

        grid_syn=$(grep -B5 "updatePresynapticKernel<<<" "$syn_file" \
            | grep "const dim3 grid" \
            | tail -1 \
            | sed -E 's/.*grid\(([0-9]+).*/\1/')

        block_syn=$(grep -B5 "updatePresynapticKernel<<<" "$syn_file" \
            | grep "const dim3 threads" \
            | tail -1 \
            | sed -E 's/.*threads\(([0-9]+).*/\1/')
    fi

    echo "$d $line_neuron $grid_neuron $block_neuron $line_syn $grid_syn $block_syn"
done