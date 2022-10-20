#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=4,5

task=fewrel
model_dir=./trained_models_hs/t5_${task}_padtomaxlenF_bmf1_adaf_DOUBLET5_bs128
dt=$(date '+%d.%m.%Y_%H.%M.%S')
mkdir $model_dir
cp ./run_fewrel_double_t5.sh $model_dir/run_fewrel_double_t5.sh_$dt
cp ./run_summarization_finetune_double_t5.py $model_dir/run_summarization_finetune_double_t5.py_$dt

python ./run_summarization_finetune_double_t5.py \
                --tokenizer_model_name_or_path t5-small \
                --model_name_or_path /home/pugachev/github/T5_KG/pretrained_models/double_t5-small_model \
                --state_dict /home/pugachev/github/T5_KG/pretrained_models/double_t5-small_state_dict \
                --triplet_model_name_or_path /home/pugachev/github/T5_KG/trained_models/t5_trex_pretrain_padtomaxlenF/checkpoint-128919 \
                --cache_dir ./downloaded_models \
                --output_dir $model_dir \
\
                --text_column input \
                --text_for_generation_column input_for_generation \
                --summary_column label \
                --task_name $task \
                --train_file /home/pugachev/github/T5_KG/data/${task}_double_t5_json/train.json \
                --validation_file /home/pugachev/github/T5_KG/data/${task}_double_t5_json/dev.json \
                --labels_file /home/pugachev/github/T5_KG/data/${task}_double_t5_json/labels.json \
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
                --pad_to_max_length False \
\
                --do_predict True \
                --load_best_model_at_end True \
		--metric_for_best_model f1_micro \
                --test_file /home/pugachev/github/T5_KG/data/${task}_double_t5_json/test.json

#--triplet_model_name_or_path /home/pugachev/github/T5_KG/trained_models_V100/t5base_trex_pretrain_padtomaxlenF_dsp_shddpsimple/checkpoint-214865 \             
# --warmup_ratio 0.3 \
# --lr_scheduler_type constant \
# --max_predict_samples 512 \
# --eval_steps 500 \
# --save_steps 500 \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

