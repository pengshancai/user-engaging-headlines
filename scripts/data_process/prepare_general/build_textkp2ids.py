import json
from collections import defaultdict



with open('../recsum_/data/newsroom/kp_7.0/test-id2textkps.json') as f:
    id2textkps = json.load(f)

textkp2id = defaultdict(list)
for idx, kps in id2textkps.items():
    kps = list(set(kps[:-1].split(';')))
    for kp in kps:
        textkp2id[kp].append(idx)

textkp2idx = dict(sorted(textkp2id.items(), key=lambda item: len(item[1]), reverse=True))

for i, (textkp, ids) in enumerate(textkp2idx.items()):
    if i % 1000 == 0:
        print('%s-%s' % (i, len(ids)))
