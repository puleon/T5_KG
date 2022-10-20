#!/bin/bash

#tasks=( "texta" "textb" "textsc" )

#for i in 0 1 2
#do
# echo "${tasks[i]}"
#done

a=$(ls /home/pugachev/github/T5_KG/data/verbalizer_output_openentity_dev_split | wc -l)

for ((i=0; i<$a; i++))
do
b=$(wc -l < /home/pugachev/github/T5_KG/data/verbalizer_output_openentity_dev_split/verb_trip_${i}.txt)
#printf -v b '%d\n' wc -l /home/pugachev/github/T5_KG/data/verbalizer_output_openentity_dev_split/verb_trip_${i}.txt
#c="$(printf '%d' ${b})"
if [[ ${b} -gt 1 ]]; then 
echo ${i}
fi
done
