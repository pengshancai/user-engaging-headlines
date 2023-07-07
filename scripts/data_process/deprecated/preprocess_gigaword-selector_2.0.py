import json
import jsonlines
import random
from tqdm import tqdm
import numpy as np
"""
A KP is compared to the entire synthesized reading history
"""
base_file = '../recsum_/data/gigaword/%s.jsonl'
kp2ids_file = '../recsum_/data/gigaword/synthesized_user/%s-textkp2idx.json'
output_file = '../recsum_/data/gigaword/selector/%s-selector-2.0_x.json'
title_file = '../recsum_/data/gigaword/synthesized_user/%s-titles.json'
NUM_NEWS_PER_USER = [15, 16, 17, 18, 19]

for split in ['train']:  # 'valid', 'test', 'train'
    with open(title_file % split, 'r') as f:
        titles_all = json.load(f)
    with open(kp2ids_file % split, 'r') as f:
        kp2ids = json.load(f)
    kps, nums = [], []
    for i, (kp, news_ids) in enumerate(kp2ids.items()):
        kps.append(kp)
        nums.append(len(news_ids))
    p = np.array(nums) / np.sum(nums)
    if split in {'valid', 'test'}:
        num_instances = 50000
    else:
        with open(base_file % split) as f:
            lines = f.readlines()
        num_instances = int(len(lines)/5)
        del lines
    progress = tqdm(range(num_instances), desc='Building synthesized users')
    data = []
    for _ in range(num_instances):
        _ = progress.update(1)
        kp = random.choices(kps, p)[0]
        cand_pool = kp2ids[kp]
        num_news = random.choice(NUM_NEWS_PER_USER)
        if len(cand_pool) < num_news:
            news_ids = cand_pool
        else:
            random.shuffle(cand_pool)

            news_ids = random.sample(cand_pool, num_news)
        history = '; '.join([titles_all[news_idx] for news_idx in news_ids])
        data.append({'kp': kp, 'history': history})
    with open(output_file % split, 'w') as f:
        info = {
            'version': 'gw-sl-2.0',
            'data': data
        }
        json.dump(info, f)


    # with jsonlines.open( % split, 'w') as f:








