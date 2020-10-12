#!/usr/bin/env bash

running_index_list=(0 1 2 3 4 5 6 7 8 9)

for running_index in "${running_index_list[@]}"; do
    mkdir -p output /
    python model.py --running_index="$running_index" > output/"$running_index".out
done

####

# The best hyperparameters are set as default in model.py. If you want to try
# different hyperparameters, pass them as below to the python command in the
# for above.

# --epoch=<num_of_epochs>
# --learning_rate=<learning_rate>
#	--batch_size=<batch_size>
#	--latent_dim=<latent dimension between the two layers of the GNN>
#	--max_node_num=<maximum number of nodes in a graph in the entire dataset>