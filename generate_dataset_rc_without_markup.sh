#!/bin/bash

python ./save_dataset_to_t5_json.py \
		--function_id 3 \
		--data_dir ./data/tacred \
		--output_dir ./data/tacred_wo_markup_json \
		--task_name tacred
