#!/bin/bash

python ./save_dataset_to_t5_json.py \
		--use_gold_triplets True \
                --triplets_dir ./data/gold_triplets \
		--function_id 3 \
		--data_dir ./data/OpenEntity \
		--output_dir ./data/openentity_with_gold_triplets_json \
		--task_name openentity
