import json
import random
from tqdm import tqdm
import numpy as np

split = 'dev'

with open('../recsum_/data/newsroom/%s-titles.json' % split, 'r') as f:
    titles_all = json.load(f)

with open('../recsum_/data/newsroom/%s-textkp2idx.json' % split, 'r') as f:
    kp2ids = json.load(f)

NUM_USERS = 50000
kps, nums = [], []
for i, (kp, news_ids) in enumerate(kp2ids.items()):
    kps.append(kp)
    nums.append(len(news_ids))

p = np.array(nums) / np.sum(nums)


def create_mixed_user(num_kps=3, num_news=15):
    kps_user = list(set(random.choices(kps, p, k=num_kps)))
    num_news_kps = random.sample(list(range(1, num_news - 1)), k=len(kps_user) - 1)
    num_news_kps.sort()
    num_news_kps = [0] + num_news_kps + [num_news]
    history = ''
    for i in range(len(kps_user)):
        kp = kps_user[i]
        cand_pool = kp2ids[kp]
        num_news_kp = num_news_kps[i + 1] - num_news_kps[i]
        if len(cand_pool) < num_news_kp:
            news_ids = cand_pool
        else:
            news_ids = random.sample(cand_pool, num_news_kp)
        history += '; '.join([titles_all[news_idx].replace(';', ',') for news_idx in news_ids]) + ';'
    return kps_user, history[:-1], num_news_kps


users = []
progress = tqdm(range(NUM_USERS))
for _ in range(NUM_USERS):
    _ = progress.update(1)
    num_kps = 1
    num_news = random.choices(list(range(12, 18)))[0]
    kps_user, history, num_news_kps = create_mixed_user(num_kps, num_news)
    users.append({'kps': kps_user, 'history': history, 'num_news_kps': num_news_kps})

with open('../recsum_/data/newsroom/%s-kp-history_1.2.json' % split, 'w') as f:
    info = {
        'version': 'kp-history-1.2',
        'data': users
    }
    json.dump(info, f)






