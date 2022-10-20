import json
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument("--input_triplets_file", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")
    parser.add_argument("--input_ids_file", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    with open(args.input_triplets_file) as f:
        data = f.readlines()
    with open(args.input_ids_file) as f:
        ids = json.load(f)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for i in range(len(ids)-1):
        with open('{}/verb_trip_{}.txt'.format(args.output_dir, i), 'w') as f:
            f.write('id\ttext\ttitle\n')
            for k, j in enumerate(range(ids[i], ids[i+1]), start=1):
                f.write("{}\t{}\t['']\n".format(k, data[j].strip()))


if __name__ == '__main__':
    main()
