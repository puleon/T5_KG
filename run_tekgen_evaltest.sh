#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9

model_dir=./trained_models/t5_trextekgen_pretrain_padtomaxlenF_adaf
dt=$(date '+%d.%m.%Y_%H.%M.%S')

mkdir $model_dir
cp ./run.sh $model_dir/run.sh_$dt
cp ./run_summarization.py $model_dir/run_summarization.py_$dt

python ./run_summarization.py \
                --model_name_or_path t5-small \
                --cache_dir ./downloaded_models \
                --resume_from_checkpoint $model_dir/checkpoint-276810 \
                --output_dir $model_dir \
\
                --text_column input \
                --summary_column label \
                --train_file "./data/trex_json/train.json,./data/tekgen_json/train.json" \
                --validation_file "./data/trex_json/dev.json" \
                --preprocessing_num_workers 20 \
                --max_source_length 256 \
                --max_target_length 64 \
                --generation_max_length 64 \
                --val_max_target_length 64 \
                --source_prefix  "" \
                --predict_with_generate True \
                --evaluation_strategy epoch \
\
                --do_train False \
                --do_eval False \
                --max_eval_samples 128 \
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 64 \
                --max_eval_samples 128 \
                --learning_rate 1e-3 \
                --optim adafactor \
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
                --test_file ./data/tekgen_json/test.json \
                --do_predict True \

# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

