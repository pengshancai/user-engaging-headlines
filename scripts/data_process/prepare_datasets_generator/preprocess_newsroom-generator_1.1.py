"""
Preprocess the dataset by adding noisy key phrase prompt
k_1.1: Unlike 1.0, 1.1 only use one KP in generation
"""
import os
import jsonlines
import json
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate newsroom dataset")
    parser.add_argument("--newsroom_original_path", type=str, default="")
    parser.add_argument("--base_path", type=str, default='../recsum_/data/newsroom/')
    args = parser.parse_args()
    return args


def process_kps(kps):
    if kps.endswith(';'):
        kps = kps[:-1]
    kps = list(set(kps.split(';')))
    return kps


def process_title(title):
    if ' : People.com' in title:
        title = title.replace(' : People.com', '')
    if '- NYTimes.com' in title:
        title = title.replace(' - NYTimes.com', '')
    return title


def process_text(text):
    text = text.replace('\n', ' ')
    return text


if __name__ == "__main__":
    args = parse_args()
    if not args.newsroom_original_path.endswith('/'):
        args.newsroom_original_path = args.newsroom_original_path + '/'
    if not args.base_path.endswith('/'):
        args.base_path = args.base_path + '/'
    for split in ['train', 'dev']:
        with open(args.base_path + 'kp_1.0/%s-url2titlekps.json' % split) as f:
            url2title_kps = json.load(f)
        progress = tqdm(range(len(url2title_kps)))
        recs = []
        os.makedirs(args.base_path + 'kp_1.1/', exist_ok=True)
        cnt = 0
        with open(args.newsroom_original_path + '%s.jsonl' % split) as json_file:
            with jsonlines.open(args.base_path + 'kp_1.1/%s.json' % split, mode='w') as writer:
                for i, line in enumerate(json_file):
                    _ = progress.update(1)
                    try:
                        info = json.loads(line)
                        url = info['url']
                        title = process_title(info['title'])
                        text = process_text(info['text'])
                        title_kps = process_kps(url2title_kps[url])
                        _ = writer.write({'text': title_kps[0] + '</s> ' + text, 'title': title})
                        cnt += 1
                    except:
                        continue

