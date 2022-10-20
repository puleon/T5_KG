#!/bin/bash

python ./save_dataset_to_t5_json.py \
		--function_id 5 \
		--data_dir ./data/OpenEntity \
		--output_dir ./data/openentity_wo_markup_json \
		--task_name openentity
