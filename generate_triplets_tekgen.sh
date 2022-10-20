#!/bin/bash

python ./wikidata_sparql_triplets.py \
                --dataset_name tekgen \
                --data_dir /home/pugachev/github/T5_KG/data/tekgen \
                --find_paired_triplets False \
                --set_names "train,dev,test" \
                --output_dir /home/pugachev/github/T5_KG/data/single_gold_triplets_tekgen

