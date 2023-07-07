import json
import pickle
import torch
from models.summarizer import load_tokenizer_user_feature
from tqdm import tqdm


def get_overlap(title, txt):
    return len(set(title.lower().split()) & set(title.lower().split())) * 1.0 / len(txt.split())


def judge_good_title(title, txt):
    if len(title.split(' ')) < 6:
        return False
    if get_overlap(title, txt) < 0.55:
        return False
    return True

pass

"""
Newsroom
"""
for split in ['dev', 'test', 'train']:
    print('Processing %s set' % split)
    print('Reading files')
    with open('../recsum_/data/newsroom/%s.jsonl' % split, 'r') as json_file:
        print('Reading content')
        recs = []
        for line in json_file:
            rec = json.loads(line)
            title = rec["title"]
            doc = rec["text"]
            recs.append({'text': doc, 'title': title})
    print('Writing files')
    if split == 'dev':
        split = 'validation'
    with open('../recsum_/data/newsroom/%s.json' % split, 'w') as f:
        info = {
            'version': 1.0,
            'data': recs
        }
        json.dump(info, f)

# from datasets import load_dataset
# dataset = load_dataset('json', data_files={'validation': '../recsum_/data/newsroom/validation.json'}, field='data')

"""
Gigaword
"""
processed_summ_datasets = {}
for split in ['valid', 'test', 'train']:
    print('Processing %s set' % split)
    print('Reading files')
    with open('../recsum_/data/gigaword/%s.jsonl' % split, 'r') as json_file:
        print('Reading content')
        recs = []
        for line in json_file:
            rec = json.loads(line)
            title = rec["headline"]
            doc = ' '.join(rec["text"])
            recs.append({'text': doc, 'title': title})
    print('Writing files')
    if split == 'valid':
        split = 'validation'
    with open('../recsum_/data/gigaword/%s.json' % split, 'w') as f:
        info = {
            'version': 1.0,
            'data': recs
        }
        json.dump(info, f)



