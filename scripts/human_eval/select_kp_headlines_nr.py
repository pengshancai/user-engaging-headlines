import json
import jsonlines
import csv
import random
from collections import defaultdict

# with open('../recsum_/data/newsroom/synthesized_user/dev-textkp2idx.json') as f:
#     textkp2idx_dev = json.load(f)

with open('../recsum_/data/newsroom/synthesized_user/test-textkp2idx.json') as f:
    textkp2idx_test = json.load(f)

with open('../recsum_/data/newsroom/synthesized_user/train-textkp2idx.json') as f:
    textkp2idx_train = json.load(f)

textkp2idx_eval = textkp2idx_test
# for textkp, ids in textkp2idx_dev.items():
#     if textkp == "":
#         continue
#     if textkp not in textkp2idx_eval:
#         textkp2idx_eval[textkp] = ids
#     else:
#         textkp2idx_eval[textkp] += ids

textkps = []
for textkp, test_ids in textkp2idx_eval.items():
    if textkp in textkp2idx_train:
        textkps.append(textkp)

textkps_selected_test = [
    'Shanghai', 'Singapore', 'Shenzhen', 'Upper East Side Manhattan', 'Hawaii', 'New Zealand', 'Las Vegas',
    'Oscars', 'Tattoo', 'Tony Awards', 'Star Wars', 'Photography', 'Broadway', 'Hip-Hop',
    'Harry Potter', 'Netflix',
    'Jeff Bezos', 'Celine Dion', 'Jimmy Kimmel', 'Brad Pitt', 'Jack Ma',
    'Global Warming', 'K-12 Education', "Alzheimer's Disease", 'Diabetes',
    'Alibaba Group', 'Microsoft Corp', 'Walt Disney Co', 'Tesla', 'Twitter',
    'Bloomberg.com', 'Lockheed Martin Corp',
    'Coffee', 'Diet', 'Fitness', 'Bicycling', 'Skiing', 'Marathon', 'Golden State Warriors', 'Jeremy Lin',
    'Kobe Bryant', 'LeBron James', 'Swimming', 'Adidas', 'Stanford University', 'Harvard', 'MIT', 'World War II',
    'GoPro', 'Huawei', 'Big Data', 'Nintendo Switch',
    'South China Sea', 'Jon Snow', 'Sushi',
    'Alexa', 'Gardening', 'Heart Attack', 'Obesity', 'KFC', 'Vampire',
]



# textkps_selected_dev = [
#     'Singapore',
#     'Gun Control', 'Halloween', 'Brexit',
#     'Samsung', 'Netflix', 'HBO', 'H&M', 'Airbus',
#     'Steve Jobs',
#     'Madison Square Garden', 'Central Park (NYC)',
#     'Nobel Prize',
#     'Sailing', 'Dinosaur', 'Fossil',
#     'Abortion', 'Divorce',
#     # 'Dating', 'Smoking', 'Android', 'Scarlett Johansson', 'Bitcoin', 'Whole Foods',
#     # 'Wind Power', 'Boston','Pregnancy','Shake Shack',
# ]

# textkps_selected = textkps_selected_dev + textkps_selected_test
textkps_selected = textkps_selected_test

textkp2ids = {}
for textkp in textkps_selected:
    ids_kp = textkp2idx_train[textkp]
    ids_kp_sel = random.sample(ids_kp, min(50, len(ids_kp)))
    textkp2ids[textkp] = ids_kp_sel

with jsonlines.open('../recsum_/data/newsroom/train.jsonl') as f:
    idx2info = {}
    for line in f:
        idx = line['url']
        headline = line['title']
        text = line['text']
        idx2info[idx] = (headline, text)


field = ['Select', 'KP', 'Headline']
with open('../recsum_/data/newsroom/human_eval/kp_headlines_pool.csv', 'w') as f:
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
textkp2ids = {}
for textkp in textkps_selected:
    ids_kp = textkp2idx_eval[textkp]
    textkp2ids[textkp] = ids_kp

with open('../recsum_/data/newsroom/kp_7.0/test-id2textkps.json') as f:
    id2textkps_test = json.load(f)

# with open('../recsum_/data/newsroom/kp_1.0/dev-url2textkps.json') as f:
#     id2textkps_dev = json.load(f)

id2textkps = id2textkps_test

with open('../recsum_/data/newsroom/test.jsonl') as f:
    con = f.readlines()
    idx2info = {}
    for i, line in enumerate(con):
        try:
            line = json.loads(line)
            idx = line['url']
            headline = line['title']
            text = line['text']
            textkps = id2textkps[idx]
            idx2info[idx] = (headline, text, textkps)
        except:
            print('Wrong format at line %s' % i)

with open('../recsum_/data/newsroom/human_eval/kp_texts_pool.json', 'w') as f:
    text_kp2info = defaultdict(list)
    for textkp, ids in textkp2ids.items():
        for idx in ids:
            if idx in idx2info:
                text_kp2info[textkp].append(idx2info[idx])
    json.dump(text_kp2info, f)
