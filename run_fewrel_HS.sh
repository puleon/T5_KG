#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=4,5,7

task=fewrel
model_dir=/home/pugachev/github/T5_KG/trained_models_hs/${task}_large_HS_hyperopt_lr_warmupratio_numtrep_bs_dropout0.2
dt=$(date '+%d.%m.%Y_%H.%M.%S')
mkdir $model_dir
cp ./run_tacred_HS.sh $model_dir/run_tacred_HS.sh_$dt
cp ./run_summarization_finetune_hyperparameter_search.py $model_dir/run_summarization_finetune_hyperparameter_search.py_$dt

python ./run_summarization_finetune_hyperparameter_search.py \
                --model_name_or_path t5-large \
                --cache_dir /home/pugachev/github/T5_KG/downloaded_models \
                --output_dir $model_dir \
                --overwrite_cache True \
\
                --text_column input \
                --summary_column label \
                --task_name $task \
                --train_file /home/pugachev/github/T5_KG/data/${task}_json/train.json \
                --validation_file /home/pugachev/github/T5_KG/data/${task}_json/dev.json \
                --labels_file /home/pugachev/github/T5_KG/data/${task}_json/labels.json \
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
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --learning_rate 1e-3 \
                --optim adafactor \
                --dropout_rate 0.2 \
                --num_train_epochs 10.0 \
                --logging_strategy steps \
                --log_level info \
                --logging_dir $model_dir \
                --logging_steps 10 \
                --logging_first_step True \
                --save_strategy no \
                --evaluation_strategy epoch \
                --pad_to_max_length False \
\
                --do_predict False \
		--metric_for_best_model tacred_f1_macro \
                --test_file /home/pugachev/github/T5_KG/data/${task}_json/test.json

# --save_strategy epoch \                
# --load_best_model_at_end True \
# --warmup_ratio 0.3 \
# --lr_scheduler_type constant \
# --max_predict_samples 512 \
# --eval_steps 500 \
# --save_steps 500 \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

