#!/bin/bash

python ./save_dataset_to_t5_json.py \
		--use_triplets False \
		--data_dir ./data/OpenEntity \
		--output_dir ./data/openentity_with_spec_tok_json \
		--task_name openentity