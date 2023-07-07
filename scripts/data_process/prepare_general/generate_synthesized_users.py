import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse
import random
import os
import jsonlines

"""
This script builds the text_kp2ids, and title_kps_all / text_kps_all
"""


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--id2text_kps_file", type=str, default='')
    parser.add_argument("--id2title_kps_file", type=str, default='')
    parser.add_argument("--data_file", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='', required=True)
    parser.add_argument("--output_test_file", type=str, default='')
    parser.add_argument("--num_synthesized_users_per_num_kps", type=int, default=2000)
    parser.add_argument("--dataset", type=str, default='nr')
    parser.add_argument("--tasks", type=str, default='1/2/3')
    parser.add_argument("--num_kps_range", type=str, default='1,2,3,4,5')
    parser.add_argument("--min_num_news", type=int, default=15)
    parser.add_argument("--max_num_news", type=int, default=21)
    args = parser.parse_args()
    args.tasks = set(args.tasks.split('/'))
    args.num_kps_range = [int(num) for num in args.num_kps_range.split(',')]
    return args


def post_process(args):
    assert args.dataset in {'nr', 'gw'}
    with open(args.id2text_kps_file) as f:  # '../recsum_/data/newsroom/kp_%s/%s-url2textkps.json' % (kp_version, split)
        idx2text_kps = json.load(f)
    # with open(args.id2title_kps_file) as f:  # '../recsum_/data/newsroom/kp_%s/%s-url2titlekps.json' % ('1.0', split)
    #     idx2title_kps = json.load(f)
    with open(args.data_file) as f:  # '../recsum_/data/newsroom/%s.jsonl' % split
        lines = f.readlines()
    idx2title = {}
    idx2text = {}
    text_kp2ids = defaultdict(list)
    progress = tqdm(range(len(lines)), desc="Generating intermediate datasets")
    for line in lines:
        _ = progress.update(1)
        try:
            info = json.loads(line)
            if args.dataset == 'nr':
                idx = info['url']
                title = info['title']
                text = info['text']
            else:
                idx = info['id']
                title = info['headline']
                text = ' '.join(info['text'])
            text_kps = idx2text_kps[idx].split(';')
            idx2title[idx] = title
            idx2text[idx] = text
            idx2text_kps[idx] = text_kps
            for text_kp in text_kps:
                text_kp2ids[text_kp].append(idx)
        except:
            continue
    # Leave out KPs that appear in less than 10 passages
    text_kp2ids_ = text_kp2ids.copy()
    for kp, ids in text_kp2ids_.items():
        if len(ids) < 10:
            _ = text_kp2ids.pop(kp)
    # Create the output_dir if it does not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    split = args.data_file.split('/')[-1].split('.')[0]
    with open(args.output_dir + '%s-idx2title.json' % split, 'w') as f:  # '../recsum_/data/newsroom/kp_%s
        json.dump(idx2title, f)
    with open(args.output_dir + '%s-idx2text.json' % split, 'w') as f:
        json.dump(idx2text, f)
    with open(args.output_dir + '%s-textkp2idx.json' % split, 'w') as f:
        json.dump(text_kp2ids, f)


def create_mixed_user(num_kps, num_news, p, idx2text, idx2title, user_kps_all, idx2title_kps, text_kp2ids):
    user_kps = list(set(random.choices(user_kps_all, p, k=num_kps)))
    num_news_kps = random.sample(list(range(1, num_news - 1)), k=len(user_kps) - 1)
    num_news_kps.sort()
    num_news_kps = [0] + num_news_kps + [num_news]
    history = ''
    user_kp_selected = random.choice(user_kps)
    idx = random.choice(text_kp2ids[user_kp_selected])
    src = idx2text[idx]
    tgt = idx2title[idx]
    # text_kps = kps_doc_all[doc_idx]
    title_kps = idx2title_kps[idx]
    for i in range(len(user_kps)):
        user_kp = user_kps[i]
        cand_pool = text_kp2ids[user_kp]
        if user_kp == user_kp_selected:
            cand_pool = [idx_ for idx_ in cand_pool if idx_ != idx]
        num_news_kp = num_news_kps[i + 1] - num_news_kps[i]
        if len(cand_pool) < num_news_kp:
            news_ids = cand_pool
        else:
            news_ids = random.sample(cand_pool, num_news_kp)
        history += '; '.join([idx2title[news_idx].replace(';', ',') for news_idx in news_ids]) + ';'
    return src, tgt, title_kps, user_kps, user_kp_selected, history[:-1], num_news_kps, idx


def generate_synthesized_users(args):
    # load in all files
    split = args.data_file.split('/')[-1].split('.')[0]
    with open(args.output_dir + '%s-idx2title.json' % split, 'r') as f:
        idx2title = json.load(f)
    with open(args.output_dir + '%s-idx2text.json' % split, 'r') as f:
        idx2text = json.load(f)
    with open(args.id2title_kps_file) as f:  # '../recsum_/data/newsroom/kp_%s/%s-url2titlekps.json' % ('1.0', split)
        idx2title_kps = json.load(f)
    with open(args.output_dir + '%s-textkp2idx.json' % split, 'r') as f:
        text_kp2ids = json.load(f)
    user_kps_all, nums = [], []
    for i, (user_kp, news_ids) in enumerate(text_kp2ids.items()):
        user_kps_all.append(user_kp)
        nums.append(len(news_ids))
    # When selecting user core KPs, select from the distribution p, so that more common KPs get select more often
    log_nums = np.log(np.array(nums))
    p = np.array(log_nums) / np.sum(log_nums)
    progress = tqdm(range(args.num_synthesized_users_per_num_kps * len(args.num_kps_range)),
                    desc='Generating synthesized users')
    with jsonlines.open(args.output_dir + 'synthesized_users.json', mode='w') as f:
        for num_kps in args.num_kps_range:
            print('Generating users with %s core KPs' % num_kps)
            for _ in range(args.num_synthesized_users_per_num_kps):
                _ = progress.update(1)
                num_news = random.choices(list(range(args.min_num_news, args.max_num_news)))[0]
                src, tgt, kps_title, kps_user, kp_user_selected, history, num_news_kps, doc_idx = create_mixed_user(num_kps, num_news, p,
                                                                                                  idx2text, idx2title,
                                                                                                  user_kps_all, idx2title_kps,
                                                                                                  text_kp2ids)
                f.write({'src': src, 'tgt': tgt, 'kps_user': kps_user, 'kp_user_selected': kp_user_selected,
                         'kps_title': kps_title, 'history': history, 'num_news_kps': num_news_kps, 'doc_idx': doc_idx})


def add_kps_doc(args):
    with open(args.id2text_kps_file, 'r') as f:
        idx2text_kps = json.load(f)
    assert args.output_test_file != ''
    with jsonlines.open(args.output_dir + 'synthesized_users.json', mode='r') as f_in:
        lines = [line for line in f_in]
    progress = tqdm(range(len(lines)), desc='Adding text KPs to synthesized users')
    with jsonlines.open(args.output_test_file, mode='w') as f_out:
        for line in lines:
            kps_doc = idx2text_kps[line['doc_idx']]
            line['kps_doc'] = kps_doc
            _ = f_out.write(line)
            _ = progress.update(1)


if __name__ == "__main__":
    args = parse_args()
    if '1' in args.tasks:
        post_process(args)
    if '2' in args.tasks:
        generate_synthesized_users(args)
    if '3' in args.tasks:
        add_kps_doc(args)



