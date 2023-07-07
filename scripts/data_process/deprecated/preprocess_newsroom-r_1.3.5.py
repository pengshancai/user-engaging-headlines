import json
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

"""
Mixed user
"""

split = 'dev'
kp_version = '2.0'

with open('../recsum_/data/newsroom/kp_%s/%s-titles.json' % (kp_version, split), 'r') as f:
    titles_all = json.load(f)

# with open('../recsum_/data/newsroom/kp_%s/%s-texts.json' % (kp_version, split), 'r') as f:
#     texts_all = json.load(f)

with open('../recsum_/data/newsroom/kp_%s/%s-textkp2idx.json' % (kp_version, split), 'r') as f:
    text_kp2ids = json.load(f)

with open('../recsum_/data/newsroom/kp_%s/%s-text-key_phrases.json' % (kp_version, split), 'r') as f:
    text_kps_all = json.load(f)

with open('../recsum_/data/newsroom/kp_%s/%s-title-key_phrases.json' % (kp_version, split), 'r') as f:
    title_kps_all = json.load(f)


pairs = []
for i, (title, text_kps_str) in enumerate(zip(titles_all, text_kps_all)):
    text_kps = text_kps_str
    for text_kp in text_kps:
        pairs.append({'title': title, 'kp': text_kp})


with open('../recsum_/data/newsroom/kp_%s/%s-kp-title_pairs_1.3.5.json' % (kp_version, split), 'w') as f:
    info = {
        'version': 'kp-history-1.3.5',
        'data': pairs,
        'note': 'kp title match'
    }
    json.dump(info, f)


