import os
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from argparse import ArgumentParser
from transformers import T5TokenizerFast, T5ForConditionalGeneration


from utils_glue import processors, preprocessor_functions


logger = logging.getLogger(__name__)


def load_and_save_examples(args, dataset_type):
    processor = processors[args.task_name]()
    logger.info("Loading examples from dataset file at %s", args.data_dir)

    examples = processor.get_train_examples(args.data_dir, dataset_type)

    relations = None
    if args.task_name in {'trex', 'fewrel'}:
        with open(os.path.join(args.data_dir, 'wikidata_relations.json')) as f:
            relations = json.load(f)

    triplets = None
    if args.use_triplets:
        with open(os.path.join(args.triplets_dir, 'generated_predictions_{}_{}.txt'.format(args.task_name, dataset_type))) as f:
            triplets = f.readlines()
        triplets = [el.strip() for el in triplets]

        assert len(examples) == len(triplets)

    t5_examples = preprocessor_functions[args.task_name][args.function_id](examples, triplets, relations)

    dataset = {"version": "0.1.0"}
    dataset["data"] = t5_examples

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, '{}.json'.format(dataset_type)), 'w') as f:
        json.dump(dataset, f)



def main():
    parser = ArgumentParser()

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--use_triplets", default=False, type=bool, required=False,
                        help="Whether to use triplets or not.")
    parser.add_argument("--function_id", default=0, type=int, required=False,
                        help="Should be 1 with use_triplets set to True.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--triplets_dir", default=None, type=str, required=False,
                        help="The triplets data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    load_and_save_examples(args, 'train')

    load_and_save_examples(args, 'dev')

    load_and_save_examples(args, 'test')


    print(1)

if __name__ == '__main__':
    main()
