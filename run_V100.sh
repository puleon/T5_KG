#!/bin/bash

#source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_dir=./trained_models/t5large_trex_pretrain_padtomaxlenF_ep10
dt=$(date '+%d.%m.%Y_%H.%M.%S')

mkdir $model_dir
cp ./run.sh $model_dir/run.sh_$dt
cp ./run_summarization.py $model_dir/run_summarization.py_$dt

python ./run_summarization.py \
		--model_name_or_path t5-large \
                --cache_dir ./downloaded_models \
		--output_dir $model_dir \
\
                --text_column input \
                --summary_column label \
                --train_file ./data/trex_json/train.json \
                --validation_file ./data/trex_json/dev.json \
                --preprocessing_num_workers 20 \
                --max_source_length 256 \
                --max_target_length 64 \
                --generation_max_length 64 \
		--val_max_target_length 64 \
                --source_prefix  "" \
                --predict_with_generate True \
\
                --do_train \
		--per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
		--learning_rate 1e-3 \
		--num_train_epochs 10.0 \
                --logging_strategy steps \
                --log_level info \
                --logging_dir $model_dir \
                --logging_steps 100 \
                --logging_first_step True \
                --save_strategy steps \
		--save_steps 1000000 \
		--pad_to_max_length False \
\
                --test_file ./data/trex_json/dev.json \
                --do_predict True \
                --max_predict_samples 512

# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

