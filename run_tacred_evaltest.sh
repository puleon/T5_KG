#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

task=tacred
model_dir=/home/pugachev/github/T5_KG/trained_models_V100/t5base_tacred_padtomaxlenF_wtripbase_adaf
dt=$(date '+%d.%m.%Y_%H.%M.%S')
mkdir $model_dir
cp ./run_${task}_evaltest.sh $model_dir/run_${task}_evaltest.sh_$dt
cp ./run_summarization_finetune.py $model_dir/run_summarization_finetune.py_$dt

for i in 1599
do
deepspeed ./run_summarization_finetune.py \
                --model_name_or_path t5-base \
                --cache_dir ./downloaded_models \
                --resume_from_checkpoint $model_dir/checkpoint-${i} \
                --output_dir $model_dir \
\
                --text_column input \
                --summary_column label \
                --task_name $task \
                --train_file ./data/${task}_with_triplets_base_json/train.json \
                --validation_file ./data/${task}_with_triplets_base_json/dev.json \
                --labels_file ./data/${task}_with_triplets_base_json/labels.json \
                --preprocessing_num_workers 20 \
                --max_source_length 256 \
                --max_target_length 64 \
                --generation_max_length 64 \
                --val_max_target_length 64 \
                --source_prefix  "" \
                --predict_with_generate True \
                --evaluation_strategy epoch \
\
                --do_train False\
                --do_eval False \
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 64 \
                --learning_rate 1e-3 \
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
                --test_file ./data/${task}_with_triplets_base_json/test.json \
                --generated_predictions_file generated_predictions_${i}.txt
                
mv $model_dir/eval_results.json $model_dir/eval_results_${i}.json
mv $model_dir/predict_results.json $model_dir/predict_results_${i}.json
done

# --load_best_model_at_end True \
# --max_predict_samples 512
# --eval_steps 500 \
# --save_steps 500 \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

