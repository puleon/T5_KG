#!/bin/bash

# source ~/envs/verbalizer/bin/activate

gpu_id=5

tasks=("openentity")
# tasks=( "figer" "fewrel" "tacred" )
# folders=( "FIGER" "fewrel" "tacred")

# for i in 0 1 2
# for i in 0
# do

# ~15 hours
# python ./wikidata_sparql_triplets_with_empty.py \
#                 --dataset_name "${tasks[i]}" \
#                 --data_dir /home/pugachev/github/T5_KG/data/"${folders[i]}" \
#                 --find_paired_triplets True \
#                 --set_names "train,dev,test" \
#                 --output_dir /home/pugachev/github/T5_KG/data/gold_triplets_tab

# python ./wikidata_sparql_triplets_with_empty.py \
#                 --dataset_name "${tasks[i]}" \
#                 --data_dir /home/pugachev/github/T5_KG/data/"${folders[i]}" \
#                 --find_paired_triplets False \
#                 --set_names "train,dev,test" \
#                 --output_dir /home/pugachev/github/T5_KG/data/single_gold_triplets_tab

# python ~/github/T5_KG/combine_triplets_for_verbalization.py \
# 		--dataset_name "${tasks[i]}" \
# 		--input_dir ~/github/T5_KG/data/single_gold_triplets_tab \
# 		--output_dir ~/github/T5_KG/data/single_gold_triplets_for_verbalization

# python ~/github/T5_KG/combine_triplets_for_verbalization.py \
# 		--dataset_name "${tasks[i]}" \
# 		--input_dir ~/github/T5_KG/data/gold_triplets_tab \
# 		--output_dir ~/github/T5_KG/data/gold_triplets_for_verbalization
 
# python ~/github/T5_KG/concatenate_triplets_for_verbalization.py \
# 		--dataset_name "${tasks[i]}" \
# 		--input_dir_single_triplets /home/pugachev/github/T5_KG/data/single_gold_triplets_for_verbalization \
# 		--input_dir_double_triplets /home/pugachev/github/T5_KG/data/gold_triplets_for_verbalization \
# 		--output_dir /home/pugachev/github/T5_KG/data/joint_gold_triplets_for_verbalization

# for dtype in train dev test
# do

# cp ~/github/T5_KG/data/joint_gold_triplets_for_verbalization/"${tasks[i]}"_${dtype}.txt ~/github/T5_KG/data/verbalizer_input/train.source 
# cp ~/github/T5_KG/data/joint_gold_triplets_for_verbalization/"${tasks[i]}"_${dtype}.txt ~/github/T5_KG/data/verbalizer_input/train.target

# cp ~/github/T5_KG/data/joint_gold_triplets_for_verbalization/"${tasks[i]}"_${dtype}.txt ~/github/T5_KG/data/verbalizer_input/"${tasks[i]}"_${dtype}.source 
# cp ~/github/T5_KG/data/joint_gold_triplets_for_verbalization/"${tasks[i]}"_${dtype}.txt ~/github/T5_KG/data/verbalizer_input/"${tasks[i]}"_${dtype}.target

#15 hours for each subset of openentity, is not parallelized on GPUs
# export CUDA_VISIBLE_DEVICES=$gpu_id;bash /home/pugachev/github/UDT-QA/Verbalizer/generate.sh \
# 		~/github/T5_KG/data/verbalizer_input \
# 		~/Verbalizer/t5_large_verbalizer_T-F_ID-T.ckpt \
# 		~/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype} \
# 		1 "${tasks[i]}"_${dtype} 10 1 0

# few minutes
# python /home/pugachev/github/UDT-QA/Verbalizer/post_processing.py \
# 		--verbalizer_outputs ~/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}/val_outputs/"${tasks[i]}"_${dtype}_predictions_rank0.txt \
# 		--verbalizer_inputs  ~/github/T5_KG/data/verbalizer_input/"${tasks[i]}"_${dtype}.source
# done
# done

# deactivate
#source ~/envs/retriever_udt/bin/activate

#for i in 0 1 2
for i in 0
do

# python /home/pugachev/github/T5_KG/write_to_file_dataset.py \
# 		--input_dir /home/pugachev/github/T5_KG/data/"${tasks[i]}"_wo_markup_json \
# 		--output_dir /home/pugachev/github/T5_KG/data/"${tasks[i]}"_wo_markup_split

# for dtype in train dev test
# do
#~15 hours
# python /home/pugachev/github/T5_KG/write_to_file.py \
# 		--input_triplets_file /home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}/val_outputs/"${tasks[i]}"_${dtype}_predictions_rank0beam_selection.txt \
# 		--input_ids_file /home/pugachev/github/T5_KG/data/joint_gold_triplets_for_verbalization/ids_"${tasks[i]}"_${dtype}.json \
# 		--output_dir /home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split
        
# a=$(ls /home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split | wc -l)

# for ((j=0; j< $a; j++))

# do

# b=$(wc -l < /home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split/verb_trip_${j}.txt)
# if [[ ${b} -gt 1 ]]; then

# echo /home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split/verb_trip_${j}.txt

# python /home/pugachev/github/UDT-QA/DPR/parse_yaml.py \
#     --input_file /home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split/verb_trip_${j}.txt

# #~20 hours
# # export CUDA_VISIBLE_DEVICES=${gpu_id}; export MKL_THREADING_LAYER="GNU"; python /home/pugachev/github/UDT-QA/DPR/generate_dense_embeddings.py model_file=/home/pugachev/github/UDT-QA/downloads/dpr_biencoder.ckpt ctx_src=dpr_wiki out_file=/home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split/verb_trip_${j}_embs batch_size=16 shard_id=0 num_shards=1 gpu_id=0 num_gpus=1

# echo /home/pugachev/github/T5_KG/data/"${tasks[i]}"_wo_markup_split/${dtype}_sample_${j}.csv

# python /home/pugachev/github/UDT-QA/DPR/parse_yaml_qa.py \
#     --input_file /home/pugachev/github/T5_KG/data/"${tasks[i]}"_wo_markup_split/${dtype}_sample_${j}.csv

# ~12 hours
# export CUDA_VISIBLE_DEVICES=${gpu_id}; python /home/pugachev/github/UDT-QA/DPR/dense_retriever.py model_file=/home/pugachev/github/UDT-QA/downloads/dpr_biencoder.ckpt qa_dataset=[custom] ctx_datatsets=[dpr_wiki] encoded_ctx_files=[/home/pugachev/github/T5_KG/data/verbalizer_output_"${tasks[i]}"_${dtype}_split/verb_trip_${j}_embs_shard0_gpu0] out_file=[/home/pugachev/github/T5_KG/data/retriever_output_"${tasks[i]}"/${dtype}_predictions_${j}.json]

# fi        
# done        
# done

# python /home/pugachev/github/UDT-QA/DPR/parse_triplet_predictions.py --input_dataset_dir /home/pugachev/github/T5_KG/data/"${tasks[i]}"_wo_markup_json --dataset_name "${tasks[i]}" --input_predictions_dir /home/pugachev/github/T5_KG/data/retriever_output_"${tasks[i]}" --output_dir /home/pugachev/github/T5_KG/data/udt_triplets

#deactivate
source ~/envs/transformers_new/bin/activate

# python ./save_dataset_to_t5_json.py \
#                 --use_gold_triplets True \
#                 --triplets_dir /home/pugachev/github/T5_KG/data/udt_triplets \
#                 --function_id 3 \
#                 --num_triplets 1 \
#                 --data_dir ./data/OpenEntity \
#                 --output_dir ./data/openentity_with_udt_triplets1_json \
#                 --task_name openentity

# python ./save_dataset_to_t5_json.py \
#                 --use_gold_triplets True \
#                 --triplets_dir /home/pugachev/github/T5_KG/data/udt_triplets \
#                 --function_id 3 \
#                 --num_triplets 2 \
#                 --data_dir ./data/OpenEntity \
#                 --output_dir ./data/openentity_with_udt_triplets2_json \
#                 --task_name openentity

python ./save_dataset_to_t5_json.py \
                --use_gold_triplets True \
                --triplets_dir /home/pugachev/github/T5_KG/data/udt_triplets \
                --function_id 3 \
                --num_triplets 3 \
                --data_dir ./data/OpenEntity \
                --output_dir ./data/openentity_with_udt_triplets3_json \
                --task_name openentity
done
