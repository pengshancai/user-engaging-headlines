import json
import os
import jsonlines
import random
from tqdm import tqdm
import numpy as np
import datetime
import multiprocessing as mp

"""
A KP is compared to the entire synthesized reading history
"""
base_file = '../recsum_/data/gigaword/%s.jsonl'
kp2ids_file = '../recsum_/data/gigaword/synthesized_user/%s-textkp2idx.json'
output_file = '../recsum_/data/gigaword/selector/%s-selector-2.0.json'
title_file = '../recsum_/data/gigaword/synthesized_user/%s-titles.json'
NUM_NEWS_PER_USER = [15, 16, 17, 18, 19, 20]
num_cores = 36
num_folds = 1


def generate_syn_users(name, num_instances_):
    data = []
    for _ in range(num_instances_):
        kp = random.choices(kps, p)[0]
        cand_pool = kp2ids[kp]
        num_news = random.choice(NUM_NEWS_PER_USER)
        if len(cand_pool) < num_news:
            news_ids = cand_pool
        else:
            random.shuffle(cand_pool)
            news_ids = cand_pool[:num_news]
        history = '; '.join([titles_all[news_idx] for news_idx in news_ids])
        data.append({'kp': kp, 'history': history})
    return {name: data}


# split = 'train'  # 'valid', 'test', 'train'
split = 'valid'
if not os.path.exists("../recsum_/data/gigaword/selector/%s-selector-2.0/" % split):
    os.mkdir("../recsum_/data/gigaword/selector/%s-selector-2.0/" % split)

with open(title_file % split, 'r') as f:
    titles_all = json.load(f)

with open(kp2ids_file % split, 'r') as f:
    kp2ids = json.load(f)

with open(base_file % split) as f:
    lines = f.readlines()

# num_instances = int(len(lines))
num_instances = 50000
num_instances_fold = int(num_instances / num_folds / num_cores)
del lines
kps, nums = [], []
for i, (kp, news_ids) in enumerate(kp2ids.items()):
    kps.append(kp)
    nums.append(len(news_ids))

log_nums = np.log(np.array(nums))
p = np.array(log_nums) / np.sum(log_nums)
progress = tqdm(range(num_folds), desc='Progress')
for fold in range(num_folds):
    start_t = datetime.datetime.now()
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(generate_syn_users, args=(name, num_instances_fold)) for name in range(num_cores)]
    results = [p.get() for p in results]
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    results_ = []
    for result in results:
        for k, v in result.items():
            results_ += v
    with open("../recsum_/data/gigaword/selector/%s-selector-2.0/21-%s.json" % (split, fold), 'w') as f:
        json.dump(results_, f)
    progress.update(1)

"""
Combine files
"""
for split in ['valid']:
    con = []
    fnames = [fname for fname in os.listdir("../recsum_/data/gigaword/selector/%s-selector-2.0/" % split) if not fname.startswith('train-selector')]
    progress = tqdm(range(len(fnames)), desc='loading files')
    for fname in fnames:
        with open("../recsum_/data/gigaword/selector/%s-selector-2.0/" % split + fname) as f:
            con_ = json.load(f)
        con = con + con_
        _ = progress.update(1)
    progress = tqdm(range(len(con)), desc='writing data')
    with jsonlines.open("../recsum_/data/gigaword/selector/%s-selector-2.0.json" % split, 'w') as f:
        for line in con:
            _ = f.write(line)
            _ = progress.update(1)


