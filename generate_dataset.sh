#!/bin/bash

python ./save_dataset_to_t5_json.py \
	         --data_dir ./data/trex-rc \
                 --output_dir ./data/trex_json \
                 --task_name trex 
