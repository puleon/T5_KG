#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=6,7

model_dir=./trained_models/t5_trex_pretrain_padtomaxlenF_dsp_fp16
dt=$(date '+%d.%m.%Y_%H.%M.%S')

mkdir $model_dir
cp ./run_deepspeed_fp16_sharded_ddp.sh $model_dir/run_deepspeed_fp16_sharded_ddp.sh_$dt
cp ./run_summarization.py $model_dir/run_summarization.py_$dt

deepspeed --master_port 60000 ./run_summarization.py \
		--fp16 True \
                --model_name_or_path t5-small \
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
                --val_max_target_length 64 \
                --source_prefix  "" \
                --predict_with_generate True \
                --evaluation_strategy epoch \
\
                --do_train \
                --do_eval \
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 64 \
                --learning_rate 1e-3 \
                --num_train_epochs 3.0 \
                --logging_strategy steps \
                --log_level info \
                --logging_dir $model_dir \
                --logging_steps 100 \
                --logging_first_step True \
                --save_strategy epoch \
                --evaluation_strategy epoch \
                --pad_to_max_length False \
\
                --load_best_model_at_end True \
                --test_file ./data/trex_json/dev.json \
                --do_predict True \
                --max_predict_samples 512

# --sharded_ddp "zero_dp_3 offload" \
# --fp16 True \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

