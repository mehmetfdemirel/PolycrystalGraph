#!/usr/bin/env bash

validation_index=8
testing_index=9
hyper_ID=20


python model.py \
	--validation_index="$validation_index" \
	--testing_index="$testing_index"\
	--hyper=$hyper_ID \
	--folder_name="$SLURM_ARRAY_TASK_ID"\
        --seed=124 >> "$hyper_ID".out


####

# The best hyperparameters are the hyperparameter set #20.If you want to try
# different hyperparameters, you can specify different "hyper_ID".
