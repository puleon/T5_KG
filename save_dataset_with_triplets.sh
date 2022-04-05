#!/bin/bash

task=figer
dname=FIGER 

python save_dataset_with_triplets_to_t5_json.py \
	--use_triplets True \
	--data_dir ./data/${dname} \
	--triplets_dir ./trained_models/t5base_trex_pretrain_padtomaxlenF_generate_triplets \
	--output_dir ./data/${task}_with_triplets_json_base \
	--task_name $task
