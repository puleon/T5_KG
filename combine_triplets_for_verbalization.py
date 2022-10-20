import json
from collections import defaultdict
import random
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="The input data dir for the dataset in json format.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    for set_type in ("train", "dev", "test"):

        with open(os.path.join(args.input_dir, '{}_{}.json'.format(args.dataset_name, set_type))) as f:
            data = json.load(f)

        data1 = dict()
        for el in data:
            dg = defaultdict(list)
            for x in el[1]:
                dg[x[:3]].append(x)
            data1[el[0]] = list(dg.values())

        data = defaultdict(list)
        for i, el in data1.items():
            for x in el:
                ids = list(range(len(x)))
                while ids:
                    triplets = []
                    k = min(len(ids), 2)
                    for j in range(k):
                        j = random.randrange(0, len(ids))
                        id = ids.pop(j)
                        triplets.append(x[id])
                    if triplets:
                        data[i].append(triplets)
            if  i not in data:                   
                data[i] = []    
        
        data1 = defaultdict(list)
        for i, el in data.items():
            for triplets in el:
                res = ''
                for j, t in enumerate(triplets):
                    if j == 0:
                        res += '<H> [title] <T> {} <H> {} <T> {}'.format(*t.split('\t'))
                    else:
                        res += ' <H> {} <T> {}'.format(*t.split('\t')[1:])
                data1[i].append(res)
            if i not in data1:
                data1[i] = []

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        ids = [0]
        with open(os.path.join(args.output_dir, '{}_{}.txt'.format(args.dataset_name, set_type)), 'w') as f:
            for i, el in sorted(data1.items(), key=lambda x: x[0]):
                ids.append(ids[-1]+len(el))
                for x in el:
                    f.write(x)
                    f.write('\n')
        with open(os.path.join(args.output_dir, 'ids_{}_{}.json'.format(args.dataset_name, set_type)), 'w') as f:
            json.dump(ids, f)


if __name__ == "__main__":
    main()