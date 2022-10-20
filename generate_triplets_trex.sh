#!/bin/bash

python ./wikidata_sparql_triplets.py \
                --dataset_name trex \
                --data_dir /home/pugachev/github/T5_KG/data/trex-rc \
                --find_paired_triplets False \
                --set_names "train,dev" \
                --output_dir /home/pugachev/github/T5_KG/data/single_gold_triplets_trex

