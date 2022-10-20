#!/bin/bash

python ./wikidata_sparql_triplets_with_empty.py \
                --dataset_name openentity \
                --data_dir /home/pugachev/github/T5_KG/data/OpenEntity \
                --find_paired_triplets False \
                --set_names "train,dev,test" \
                --output_dir /home/pugachev/github/T5_KG/data/single_gold_triplets_tab

