#!/bin/bash

python ./save_dataset_to_t5_json.py \
		--use_triplets True \
		--triplets_dir ./trained_models/t5base_trex_pretrain_padtomaxlenF_dsp_shddpsimple_TRIPLETS \
		--function_id 1 \
		--data_dir ./data/tacred \
		--output_dir ./data/tacred_with_triplets_base_json \
		--task_name tacred

# --data_dir ./data/trex-rc \
# --output_dir ./data/trex_json \
# --task_name trex
