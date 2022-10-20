import json
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for set_name in ('train', 'dev', 'test'):
        with open(os.path.join(args.input_dir, '{}.json'.format(set_name))) as f:
            data = json.load(f)
            data = data['data']

        for i in range(len(data)):
            with open(os.path.join(args.output_dir, '{}_sample_{}.csv'.format(set_name, i)), 'w') as f:
                f.write("{}\t['']\n".format(data[i]['input']))

if __name__ == '__main__':
    main()
