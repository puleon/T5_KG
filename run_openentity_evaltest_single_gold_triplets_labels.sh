#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

task=openentity
model_dir=./trained_models/t5_openentity_padtomaxlenF_wtripsinggoldlab_bmf1_adaf
dt=$(date '+%d.%m.%Y_%H.%M.%S')
mkdir $model_dir
cp ./run_${task}.sh $model_dir/run_${task}.sh_$dt
cp ./run_summarization_finetune.py $model_dir/run_summarization_finetune.py_$dt

for i in 320
do
deepspeed ./run_summarization_finetune.py \
                --use_triplets True \
                --model_name_or_path t5-small \
                --cache_dir ./downloaded_models \
                --resume_from_checkpoint $model_dir/checkpoint-${i} \
                --output_dir $model_dir \
\
                --text_column input \
                --summary_column label \
                --task_name $task \
                --train_file ./data/${task}_with_single_gold_triplets_labels_json/train.json \
                --validation_file ./data/${task}_with_single_gold_triplets_labels_json/dev.json \
                --labels_file ./data/${task}_with_single_gold_triplets_labels_json/labels.json \
                --preprocessing_num_workers 20 \
                --max_source_length 256 \
                --max_target_length 256 \
                --generation_max_length 256 \
                --val_max_target_length 256 \
                --source_prefix  "" \
                --predict_with_generate True \
                --generation_num_beams 5 \
\
                --do_train False\
                --do_eval False \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
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
                --load_best_model_at_end True \
                --test_file ./data/${task}_with_single_gold_triplets_labels_json/test.json \
                --generated_predictions_file generated_predictions_${i}.txt
                
mv $model_dir/eval_results.json $model_dir/eval_results_${i}.json
mv $model_dir/predict_results.json $model_dir/predict_results_${i}.json
done

# --max_predict_samples 512
# --eval_steps 500 \
# --save_steps 500 \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

