import json
from collections import defaultdict
import random
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--input_dir_double_triplets", default=None, type=str, required=True,
                        help="The input data dir for the dataset in json format.")
    parser.add_argument("--input_dir_single_triplets", default=None, type=str, required=True,
                        help="The input data dir for the dataset in json format.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    for set_type in ("train", "dev", "test"):
    # for set_type in ("train", "test"):

        with open(os.path.join(args.input_dir_double_triplets, '{}_{}.txt'.format(args.dataset_name, set_type))) as f:
            data_double = f.readlines()
        with open(os.path.join(args.input_dir_double_triplets, 'ids_{}_{}.json'.format(args.dataset_name, set_type))) as f:
            ids_double = json.load(f)

        with open(os.path.join(args.input_dir_single_triplets, '{}_{}.txt'.format(args.dataset_name, set_type))) as f:
            data_single = f.readlines()
        with open(os.path.join(args.input_dir_single_triplets, 'ids_{}_{}.json'.format(args.dataset_name, set_type))) as f:
            ids_single = json.load(f)

        assert len(ids_double) == len(ids_single), 'Ids do not coincide.'
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        ids = [0]
        with open(os.path.join(args.output_dir, '{}_{}.txt'.format(args.dataset_name, set_type)), 'w') as f:
            for i in range(len(ids_double)-1):
                for j in range(ids_double[i], ids_double[i+1]):
                    f.write(data_double[j])
                for j in range(ids_single[i], ids_single[i+1]):
                    f.write(data_single[j])
                ids.append(ids_single[i+1]+ids_double[i+1])
        
        with open(os.path.join(args.output_dir, 'ids_{}_{}.json'.format(args.dataset_name, set_type)), 'w') as f:
            json.dump(ids, f)



if __name__ == "__main__":
    main()