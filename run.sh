#!/usr/bin/env bash

running_index_list=(0 1 2 3 4 5 6 7 8 9)

for running_index in "${running_index_list[@]}"; do
    mkdir -p output/
    python model.py --running_index="$running_index" > output/"$running_index".out
done
