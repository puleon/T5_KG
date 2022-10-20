#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=4,5

task=fewrel
model_dir=./trained_models/t5_${task}_padtomaxlenT_adaf_MASKEDT5
dt=$(date '+%d.%m.%Y_%H.%M.%S')
mkdir $model_dir
cp ./run_${task}_masked_t5.sh $model_dir/run_${task}_masked_t5.sh_$dt
cp ./run_summarization_finetune_1_masked.py $model_dir/run_summarization_finetune_1_masked.py_$dt

python ./run_summarization_finetune_1_masked.py \
                --model_type masked_t5 \
                --model_name_or_path t5-small\
                --cache_dir ./downloaded_models \
                --output_dir $model_dir \
\
                --text_column input \
                --summary_column label \
                --task_name $task \
                --train_file ./data/${task}_json/train.json \
                --validation_file ./data/${task}_json/dev.json \
                --labels_file ./data/${task}_json/labels.json \
                --preprocessing_num_workers 20 \
                --max_source_length 256 \
                --max_target_length 64 \
                --generation_max_length 64 \
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
                --optim adafactor \
                --num_train_epochs 10.0 \
                --logging_strategy steps \
                --log_level info \
                --logging_dir $model_dir \
                --logging_steps 10 \
                --logging_first_step True \
                --save_strategy epoch \
                --evaluation_strategy epoch \
                --pad_to_max_length True \
\
                --do_predict True \
                --load_best_model_at_end True \
                --metric_for_best_model f1_macro \
                --test_file ./data/${task}_json/test.json
                

# --lr_scheduler_type constant \
# --max_predict_samples 512
# --eval_steps 500 \
# --save_steps 500 \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

