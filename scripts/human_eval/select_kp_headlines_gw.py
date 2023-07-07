import json
import jsonlines
import csv
import random
from collections import defaultdict
from tqdm import tqdm

REBUILD = False
if REBUILD:
    with open('../recsum_/data/gigaword/synthesized_user/test-textkp2idx.json') as f:
        textkp2idx_test = json.load(f)
    with open('../recsum_/data/gigaword/kp_1.0/train-id2textkps.json') as f:
        id2textkps_train = json.load(f)
    textkp2idx_train = defaultdict(list)
    progress = tqdm(range(len(id2textkps_train)))
    for i, (idx, kps_str) in enumerate(id2textkps_train.items()):
        kps = kps_str.split(';')
        _ = progress.update(1)
        for kp in kps:
            if kp in textkp2idx_test:
                textkp2idx_train[kp].append(idx)
    with open("../recsum_/data/gigaword/synthesized_user/train-textkp2idx.json", 'w') as f:
        json.dump(textkp2idx_train, f)
else:
    with open("../recsum_/data/gigaword/synthesized_user/train-textkp2idx.json") as f:
        textkp2idx_train = json.load(f)

textkps = list(textkp2idx_train.keys())


textkps_selected = \
    ['Inflation', 'Sculpture', 'Jazz', 'Tea', 'Bermuda', 'Mars', 'Cheese', 'Beer',
     'Celine Dion', 'Brad Pitt',
     'Shanghai', 'Singapore', 'Shenzhen', 'Hawaii', 'New Zealand', 'Las Vegas',
     'Oscars', 'Tattoo', 'Tony Awards', 'Star Wars',
     'Photography', 'Broadway', 'Hip-Hop', 'Harry Potter',
     'Global Warming', 'K-12 Education',
     "Alzheimer's Disease", 'Diabetes', 'Microsoft Corp',
     'Walt Disney Co', 'Twitter', 'Lockheed Martin Corp',
     'Coffee', 'Diet', 'Fitness', 'Bicycling', 'Skiing',
     'Marathon', 'Golden State Warriors',
     'Kobe Bryant', 'LeBron James', 'Swimming',
     'Stanford University', 'Harvard', 'MIT',
     'World War II',
     'South China Sea', 'Sushi',
     'Heart Attack', 'Obesity', 'KFC',
     'Vampire'
     ]



textkp2ids = {}
for textkp in textkps_selected:
    ids_kp = textkp2idx_train[textkp]
    ids_kp_sel = random.sample(ids_kp, min(50, len(ids_kp)))
    textkp2ids[textkp] = ids_kp_sel

selected_ids = []
for _, ids in textkp2ids.items():
    selected_ids += ids

selected_ids = set(selected_ids)

with jsonlines.open('../recsum_/data/gigaword/train.jsonl') as f:
    idx2info = {}
    for line in f:
        idx = line['id']
        if idx in selected_ids:
            headline = line['headline']
            text = ' '.join(line['text'])
            idx2info[idx] = (headline, text)

field = ['Select', 'KP', 'Headline']
with open('../recsum_/data/gigaword/human_eval/kp_headlines_pool_gw.csv', 'w') as f:
    csvwriter = csv.writer(f)
    _ = csvwriter.writerow(field)
    for textkp, ids in textkp2ids.items():
        ids_selected = random.sample(ids, min(30, len(ids)))
        for idx in ids_selected:
            headline, text = idx2info[idx]
            _ = csvwriter.writerow(['', textkp, headline])

"""
Obtain headline / text from test set
"""
with open('../recsum_/data/gigaword/synthesized_user/test-textkp2idx.json') as f:
    textkp2idx_test = json.load(f)

textkp2ids = {}
for textkp in textkps_selected:
    ids_kp = textkp2idx_test[textkp]
    textkp2ids[textkp] = ids_kp

with open('../recsum_/data/gigaword/kp_1.0/test-id2textkps.json') as f:
    id2textkps_test = json.load(f)

# with open('../recsum_/data/newsroom/kp_1.0/dev-url2textkps.json') as f:
#     id2textkps_dev = json.load(f)

# id2textkps = id2textkps_test

with open('../recsum_/data/gigaword/test.jsonl') as f:
    con = f.readlines()
    idx2info = {}
    for i, line in enumerate(con):
        try:
            line = json.loads(line)
            idx = line['id']
            headline = line['headline']
            text = ' '.join(line['text'])
            textkps = id2textkps_test[idx]
            idx2info[idx] = (headline, text, textkps)
        except:
            print('Wrong format at line %s' % i)

with open('../recsum_/data/gigaword/human_eval/kp_texts_pool_gw.json', 'w') as f:
    text_kp2info = defaultdict(list)
    for textkp, ids in textkp2ids.items():
        for idx in ids:
            if idx in idx2info:
                text_kp2info[textkp].append(idx2info[idx])
    json.dump(text_kp2info, f)
