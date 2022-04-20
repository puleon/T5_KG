#!/bin/bash

python ./save_dataset_to_t5_json.py \
		--use_triplets True \
                --triplets_dir ./trained_models/t5_trex_pretrain_padtomaxlenF_generate_triplets \
		--function_id 1 \
		--data_dir ./data/fewrel \
		--output_dir ./data/fewrel_with_triplets_json \
		--task_name fewrel
