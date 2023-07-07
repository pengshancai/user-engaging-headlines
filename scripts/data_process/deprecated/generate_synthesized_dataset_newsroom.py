import json
import random
from tqdm import tqdm
import numpy as np

split = 'dev'
kp_version = '7.0'
BASE_PATH = '../recsum_/'

with open(BASE_PATH + 'data/newsroom/kp_%s/%s-titles.json' % (kp_version, split), 'r') as f:
    titles_all = json.load(f)

with open(BASE_PATH + 'data/newsroom/kp_%s/%s-texts.json' % (kp_version, split), 'r') as f:
    texts_all = json.load(f)

with open(BASE_PATH + 'data/newsroom/kp_%s/%s-textkp2idx.json' % (kp_version, split), 'r') as f:
    text_kp2ids = json.load(f)

with open(BASE_PATH + 'data/newsroom/kp_%s/%s-text-key_phrases.json' % (kp_version, split), 'r') as f:
    kps_doc_all = json.load(f)

with open(BASE_PATH + 'data/newsroom/kp_%s/%s-title-key_phrases.json' % (kp_version, split), 'r') as f:
    kps_title_all = json.load(f)


NUM_USERS = 10000
user_kps_all, nums = [], []
for i, (user_kp, news_ids) in enumerate(text_kp2ids.items()):
    user_kps_all.append(user_kp)
    nums.append(len(news_ids))

p = np.array(nums) / np.sum(nums)


def create_synthesized_users(num_kps=3, num_news=15):
    user_kps = list(set(random.choices(user_kps_all, p, k=num_kps)))
    num_news_kps = random.sample(list(range(1, num_news - 1)), k=len(user_kps) - 1)
    num_news_kps.sort()
    num_news_kps = [0] + num_news_kps + [num_news]
    history = ''
    user_kp_selected = random.choice(user_kps)
    doc_idx = random.choice(text_kp2ids[user_kp_selected])
    src = texts_all[doc_idx]
    tgt = titles_all[doc_idx]
    text_kps = kps_doc_all[doc_idx]
    title_kps = kps_title_all[doc_idx]
    for i in range(len(user_kps)):
        user_kp = user_kps[i]
        cand_pool = text_kp2ids[user_kp]
        if user_kp == user_kp_selected:
            cand_pool = [idx for idx in cand_pool if idx != doc_idx]
        num_news_kp = num_news_kps[i + 1] - num_news_kps[i]
        if len(cand_pool) < num_news_kp:
            news_ids = cand_pool
        else:
            news_ids = random.sample(cand_pool, num_news_kp)
        history += '; '.join([titles_all[news_idx].replace(';', ',') for news_idx in news_ids]) + ';'
    return src, tgt, text_kps, title_kps, user_kp_selected, history[:-1], num_news_kps


users = []
progress = tqdm(range(NUM_USERS))
for _ in range(NUM_USERS):
    _ = progress.update(1)
    num_kps = random.choices(list(range(2, 5)))[0]
    num_news = random.choices(list(range(12, 18)))[0]
    src, tgt, kps_doc, kps_title, kps_user, history, num_news_kps = create_synthesized_users(num_kps, num_news)
    users.append({'src': src, 'tgt': tgt, 'kps_doc': kps_doc, 'kps_title': kps_title, 'kps_user': kps_user, 'history': history, 'num_news_kps': num_news_kps})

with open('../recsum_/data/newsroom/kp_%s/%s-kp-history_1.3.1.json' % (kp_version, split), 'w') as f:
    info = {
        'version': 'kp-history-1.3.1',
        'data': users,
        'note': 'mixed user'
    }
    json.dump(info, f)

