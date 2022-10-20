import os
import json
from argparse import ArgumentParser

def load_and_save_examples(args, dataset_type):
    with open(os.path.join(args.data_for_generation_dir, '{}.json'.format(dataset_type))) as f:
        dataset_for_generation = json.load(f)

    with open(os.path.join(args.data_dir, '{}.json'.format(dataset_type))) as f:
        dataset_main = json.load(f)

    dataset_for_generation = dataset_for_generation["data"]
    dataset_main = dataset_main["data"]

    assert len(dataset_for_generation) == len(dataset_main)

    examples = []
    for i in range(len(dataset_main)):
        ex = dataset_main[i]
        ex['input_for_generation'] = dataset_for_generation[i]['input']
        examples.append(ex)

    dataset = {"version": "0.1.0"}
    dataset["data"] = examples

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, '{}.json'.format(dataset_type)), 'w') as f:
        json.dump(dataset, f)


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_for_generation_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    for set_name in ['train', 'dev', 'test']:
        load_and_save_examples(args, set_name)

    print(1)

if __name__ == '__main__':
    main()