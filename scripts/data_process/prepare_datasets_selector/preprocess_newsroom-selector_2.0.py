import json
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
"""
A KP is compared to the entire synthesized reading history
"""

# Build synthesized users
split = 'train'
with open('../recsum_/data/newsroom/%s-titles.json' % split, 'r') as f:
    titles_all = json.load(f)

with open('../recsum_/data/newsroom/kp_1.0/%s-textkp2idx.json' % split, 'r') as f:
    kp2ids = json.load(f)

NUM_NEWS_PER_USER = 15
NUM_USERS = 10000
kps, nums = [], []
for i, (kp, news_ids) in enumerate(kp2ids.items()):
    kps.append(kp)
    nums.append(len(news_ids))

p = np.array(nums) / np.sum(nums)
users = []
progress = tqdm(range(NUM_USERS))
print(' Building synthesized users')
for _ in range(NUM_USERS):
    _ = progress.update(1)
    kp = random.choices(kps, p)[0]
    cand_pool = kp2ids[kp]
    if len(cand_pool) < NUM_NEWS_PER_USER:
        news_ids = cand_pool
    else:
        news_ids = random.sample(cand_pool, NUM_NEWS_PER_USER)
    history = '; '.join([titles_all[news_idx] for news_idx in news_ids])
    users.append({'kp': kp, 'history': history})

# with open('../recsum_/data/newsroom/%s-kp-history_2.0.json' % split, 'w') as f:
#     info = {
#         'version': 'kp-history-2.0',
#         'data': users
#     }
#     json.dump(info, f)






