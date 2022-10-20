import json
import os
import random
from collections import defaultdict
from argparse import ArgumentParser


#number of lines in files
# single_gold_triplets_tekgen 6135096
# single_gold_triplets_tekgen_old 4259510
# single_gold_triplets_trex 5270670
# single_gold_triplets_trex_old 3598707

# gold_triplets_trex 5023004
# gold_triplets_trex_old 4722452
# gold_triplets_tekgen 4882232
# gold_triplets_tekgen_old 4805175

# Как берутся отрицательные сэмплы из набора отрицательных
# Как указать, что hard_negative не нужно брать
# Зачем нужны answers, что туда подставить
# Сколько обучается
# отделить dev файл, что если там будет не одинаковое число кандидатов


def main():
    parser = ArgumentParser()

    parser.add_argument("--input_triplets_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")
    parser.add_argument("--input_single_triplets_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")
    parser.add_argument("--input_dataset_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    triplets = defaultdict(set)
    for i in range(75):
        with open(os.path.join(args.input_triplets_dir, 'trex_train_{}.json'.format(i))) as f:
            for el in f.readlines():
                t = json.loads(el)
                index = t[0]
                t = set([' '.join(el.split('\t')) for el in t[1]])
                triplets[index] &= t

    for i in range(75):
        with open(os.path.join(args.input_single_triplets_dir, 'trex_train_{}.json'.format(i))) as f:
            for el in f.readlines():
                t = json.loads(el)
                index = t[0]
                t = set([' '.join(el.split('\t')) for el in t[1]])
                triplets[index] &= t

    with open(os.path.join(args.input_dataset_dir, 'train.json')) as f:
        data = json.load(f)
    data = data['data']

    res = []
    for i in range(len(data)):
        if len(triplets[i]) > 0:
            sample = dict()
            sample['question'] = data[i]['input']
            sample['answers'] = ['']
            sample['positive_ctxs'] = [{'title': '', 'text': data[i]['label']}]
            neg_ctxs = triplets[i] - set([data[i]['input']])
            sample['negative_ctxs'] = [{'title': '', 'text': el} for el in neg_ctxs]
            sample['negative_ctxs'] = []
        res.append(sample)

    random.shuffle(res)

    train, dev = res[:int(0.9*len(res))], res[int(0.9*len(res)):]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, 'train.json'), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(args.output_dir, 'dev.json'), 'w') as f:
        json.dump(dev, f)



if __name__ == '__main__':
    main()
