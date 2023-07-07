import json
import jsonlines
import numpy as np
from collections import defaultdict

"""
article / headline length
"""
info = defaultdict(list)
with jsonlines.open('../recsum_/data/gigaword/kp_1.1/valid.jsonl') as f:
    for line in f:
        text, title = line['text'], line['title']
        info['len_text'].append(len(text.split()))
        info['len_title'].append(len(title.split()))

for key, values in info.items():
    print('%s:\t%s' % (key, np.mean(values)))

cnt = 0
with jsonlines.open('../recsum_/data/gigaword/kp_1.1/valid.jsonl') as f:
    for line in f:
        cnt += 1

"""
Number of KPs per article
"""
# Gigaword
with open('../recsum_/data/gigaword/kp_1.0/test-id2textkps.json') as f:
    id2textkps = json.load(f)

ls = []
for idx, textkps in id2textkps.items():
    ls.append(len(textkps.split(';')))

print(np.mean(ls))   # train: 10.814339645957231; test: 10.82689183692215

# Newsroom
with open('../recsum_/data/newsroom/kp_1.0/dev-url2textkps.json') as f:
    id2textkps = json.load(f)

ls = []
for idx, textkps in id2textkps.items():
    ls.append(len(textkps.split(';')))

print(np.mean(ls))  # train: 11.367434946929729


"""
Total number of KPs
"""
# Gigaword
with open('../recsum_/data/gigaword/synthesized_user/train-textkp2idx.json') as f:
    textkp2idx_train_ = json.load(f)

with open('../recsum_/data/gigaword/synthesized_user/test-textkp2idx.json') as f:
    textkp2idx_test_ = json.load(f)

kps = []
for kp, ids in textkp2idx_train_.items():
    if len(ids) > 10:
        kps.append(kp)

print(len(kps)) # 25,084

# Newsroom
with open('../recsum_/data/newsroom/synthesized_user/train-textkp2idx.json') as f:
    textkp2idx_train = json.load(f)

with open('../recsum_/data/newsroom/synthesized_user/dev-textkp2idx.json') as f:
    textkp2idx_dev = json.load(f)

with open('../recsum_/data/newsroom/synthesized_user/test-textkp2idx.json') as f:
    textkp2idx_test = json.load(f)

kps = []
for kp, ids in textkp2idx_train.items():
    if len(ids) > 10:
        kps.append(kp)

print(len(kps)) # 48,820

"""
Test set statistics
"""
with jsonlines.open('../recsum_/data/gigaword/synthesized_user/test.json') as f:
    con = [line for line in f]


"""
Selector statistics
"""
history_len = []
path = "../recsum_/data/gigaword/selector/train-selector-2.0.json"
# path = "../recsum_/data/newsroom/selector/dev-selector-2.1.json"
with jsonlines.open(path) as f:
    for i, line in enumerate(f):
        num_articles = len(line['history'].split('; '))
        history_len.append(num_articles)
        if i % 100000 == 0:
            print(i)

print(np.mean(history_len))


