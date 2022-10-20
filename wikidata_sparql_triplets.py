import os
import time
import json
import requests
from argparse import ArgumentParser
from functools import partial



def find_label(ent):
    url = 'https://query.wikidata.org/sparql'
    query = '''
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    SELECT  *
    WHERE {
            wd:%s rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
          } 
    LIMIT 1''' % (ent,)
    r = requests.get(url, params = {'format': 'json', 'query': query}, headers = {'User-agent': 'your bot 0.53'})
    while r.status_code == 429:
        time.sleep(2)
        r = requests.get(url, params = {'format': 'json', 'query': query}, headers = {'User-agent': 'your bot 0.53'})
    wdata = r.json()
    if wdata['results']['bindings']:
        return wdata['results']['bindings'][0]['label']['value']
    else:
        return None


def find_relation(ent1, ent2):
    url = 'https://query.wikidata.org/sparql'
    query = '''
    SELECT ?p ?pLabel WHERE {
       wd:%s ?prop wd:%s .
       ?p wikibase:directClaim ?prop .
       SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
       }
     }''' % (ent1, ent2)
    r = requests.get(url, params = {'format': 'json', 'query': query}, headers = {'User-agent': 'your bot 0.53'})
    while r.status_code == 429:
        time.sleep(2)
        r = requests.get(url, params = {'format': 'json', 'query': query}, headers = {'User-agent': 'your bot 0.53'})

    wdata = r.json()
    if wdata['results']['bindings']:
        return wdata['results']['bindings'][0]['pLabel']['value']
    else:
        return None


def get_entities_openentity(x):
    x = x['ents']
    if not x:
        return x
    x.sort(key=lambda x: x[3], reverse=True)
    x = [el[0] for el in x]
    entities_labels = {}
    for el in x:
        label = find_label(el)
        if label:
            entities_labels[el] = label    
    return entities_labels


def get_entities_trex(x):
    entities_labels = {}
    if x['subj_label'].startswith('Q'):
        entities_labels[x['subj_label']] = ' '.join(x['token'][x['subj_start']: x['subj_end']+1])
    if x['obj_label'].startswith('Q'):    
        entities_labels[x['obj_label']] = ' '.join(x['token'][x['obj_start']: x['obj_end']+1])
    return entities_labels


def get_entities_tekgen(x, entities):
    entities_labels = {}
    for t in x['triples']:
        if len(t) == 3:
            ent1, _, ent2 = t
            if ent1 in entities:
                entities_labels[entities[ent1]] = ent1   
            if ent2 in entities:
                entities_labels[entities[ent2]] = ent2   
    return entities_labels


def find_triplets(entities_labels):
    if len(entities_labels) < 2:
        return []

    triplets = []
    for ent1 in entities_labels:
        for ent2 in entities_labels:
            if ent1 != ent2:
                rel = find_relation(ent1, ent2)
                if rel:
                    triplets.append('{}\t{}\t{}'.format(entities_labels[ent1], rel, entities_labels[ent2]))
    return triplets


def find_relation_object(ent):
    url = 'https://query.wikidata.org/sparql'
    query = '''
    SELECT ?p ?pLabel ?o ?oLabel WHERE {
        wd:%s ?prop ?o .
        ?p wikibase:directClaim ?prop .
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en" .
        }
    }''' % (ent,)
    r = requests.get(url, params = {'format': 'json', 'query': query}, headers = {'User-agent': 'your bot 0.53'})
    while r.status_code == 429:
        time.sleep(2)
        r = requests.get(url, params = {'format': 'json', 'query': query}, headers = {'User-agent': 'your bot 0.53'})
    wdata = r.json()
    po = []
    if wdata['results']['bindings']:
        for el in wdata['results']['bindings']:
            if 'xml:lang' in el['oLabel'] and el['oLabel']['xml:lang'] == 'en' and not el['oLabel']['value'].startswith('Category:'):
                po.append([el['pLabel']['value'], el['oLabel']['value']])
        return po
    else:
        return None

def find_single_triplets(entities_labels):
    triplets = []
    for ent in entities_labels:
            ros = find_relation_object(ent)
            if ros:
                for ro in ros: 
                    triplets.append('{}\t{}\t{}'.format(entities_labels[ent], *ro))
    return triplets

def main():
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--set_names", default="train,dev,test", type=str, required=False,
                        help="The name of the task to train.")
    parser.add_argument("--find_paired_triplets", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to use triplets or not.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir for the dataset in json format.")

    args = parser.parse_args()

    if args.dataset_name == 'tekgen':
        entities = {}
        with open('/home/pugachev/github/T5_KG/data/tekgen/entities.jsonl') as f:
            for l in f.readlines():
                x = json.loads(l.strip())
                entities[x['name']] = x['id'] 

    triplet_func = find_triplets if args.find_paired_triplets else find_single_triplets
    entity_func = None
    if args.dataset_name == 'openentity':
        entity_func = get_entities_openentity
    elif args.dataset_name == 'trex':
        entity_func = get_entities_trex
    elif args.dataset_name == 'tekgen':
        entity_func = partial(get_entities_tekgen, entities=entities)

    for set_name in args.set_names.split(','):
        print(set_name, '\n***')

        if args.dataset_name in {'openentity', 'trex'}:
            with open(os.path.join(args.data_dir, '{}.json'.format(set_name))) as f:
                data = json.load(f)
        elif args.dataset_name == 'tekgen':
            data = []
            set_name = 'validation' if set_name == 'dev' else set_name
            with open(os.path.join(args.data_dir, 'quadruples-{}.tsv'.format(set_name))) as f:
                for l in f.readlines():
                    data.append(json.loads(l.strip()))

        count = 0
        count_num = 0
        triplets = []
        for i, el in enumerate(data):
            entities_labels = entity_func(el)
            t = triplet_func(entities_labels)
            if t:
                print(i, t)
                triplets.append((i, t))
                count += 1
                count_num += len(t)

        print(count/len(data))
        print(count_num/len(data))


        with open(os.path.join(args.output_dir, '{}_{}.json'.format(args.dataset_name, set_name)), 'w') as f:
            json.dump(triplets, f)

if __name__ == "__main__":
    main()    