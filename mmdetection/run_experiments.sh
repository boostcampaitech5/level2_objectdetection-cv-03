#!/bin/bash

# Specify experiment configurations
experiments=(
    "BashTest1 test1 True"
    "BashTest2 test2 False"
    "BashTest3 test3 True"
    # Add more experiment configurations here...
)
# Iterate over experiments
for experiment in "${experiments[@]}"; do
    # Split experiment configuration into array
    IFS=' ' read -r -a config <<< "$experiment"
    
    # Run experiment
    python tools/train.py "${config[0]}" --train-type "${config[1]}" --fp16 "${config[2]}"
done

