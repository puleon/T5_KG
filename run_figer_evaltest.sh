#!/bin/bash

source ~/envs/transformers_new/bin/activate

export CUDA_VISIBLE_DEVICES=6,7

task=figer
model_dir=./trained_models/t5_${task}_padtomaxlenF_resume
dt=$(date '+%d.%m.%Y_%H.%M.%S')
mkdir $model_dir
cp ./run_figer.sh $model_dir/run_figer.sh_$dt
cp ./run_summarization_finetune.py $model_dir/run_summarization_finetune.py_$dt

for i in 30000 60000 90000 120000 150000
do
python ./run_summarization_finetune.py \
                --model_name_or_path t5-small \
                --cache_dir ./downloaded_models \
                --resume_from_checkpoint $model_dir/checkpoint-${i} \
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
		--max_target_length 256 \
                --generation_max_length 64 \
                --val_max_target_length 64 \
                --source_prefix  "" \
                --predict_with_generate True \
                --evaluation_strategy epoch \
\
                --do_train False \
                --do_eval False\
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 64 \
                --learning_rate 1e-3 \
                --num_train_epochs 5.0 \
                --logging_strategy steps \
                --log_level info \
                --logging_dir $model_dir \
                --logging_steps 1500 \
                --logging_first_step True \
                --save_strategy steps \
                --save_steps 7500 \
                --evaluation_strategy steps \
                --eval_steps 7500 \
                --pad_to_max_length False \
\
                --do_predict True \
                --load_best_model_at_end True \
                --test_file ./data/${task}_json/test.json \
                --generated_predictions_file generated_predictions_${i}.txt

mv $model_dir/eval_results.json $model_dir/eval_results_${i}.json
mv $model_dir/predict_results.json $model_dir/predict_results_${i}.json
done

#--max_target_length 64 \
#--val_max_target_length 64 \
#--generation_max_length 64 \

# --max_predict_samples 512
# --eval_steps 500 \
# --save_steps 500 \
# --overwrite_output_dir True \
# --overwrite_cache True \
# --max_train_samples 512 \
# --max_eval_samples 512 \

