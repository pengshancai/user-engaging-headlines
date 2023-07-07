"""
This script generates base datasets
"""

import os.path
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from nltk import sent_tokenize
from tqdm import tqdm
import torch
from collections import defaultdict
import argparse

base_path = '/data/home/pengshancai/workspace/recsum_/data/newsroom/'
data_path_in = base_path + '%s.jsonl'  # The original newsroom jsonl file
device = torch.device('cuda')

# # kp_1.0 setting
# NUM_FOLDS = 1
# NUM_WORDS_PER_FOLD = 400
# MIN_LENGTH = 50
# MAX_LENGTH = 100

# # kp_2.0 setting
# NUM_FOLDS = 4
# NUM_WORDS_PER_FOLD = 100
# MIN_LENGTH = 50
# MAX_LENGTH = 100

# # kp_3.0 setting
# NUM_FOLDS = 2
# NUM_WORDS_PER_FOLD = 200
# MIN_LENGTH = 50
# MAX_LENGTH = 100

# # kp_5.0 setting
# NUM_FOLDS = 2
# NUM_WORDS_PER_FOLD = 200
# MIN_LENGTH = 35
# MAX_LENGTH = 200

# # kp_6.0 setting
# NUM_FOLDS = 1
# NUM_WORDS_PER_FOLD = 200
# MIN_LENGTH = 70
# MAX_LENGTH = 100

# # kp_7.0 setting (Final setting)
NUM_FOLDS = 1
NUM_WORDS_PER_FOLD = 400
MIN_LENGTH = 60
MAX_LENGTH = 100
DO_SAMPLE = True
tokenizer = BartTokenizer.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes')
model = BartForConditionalGeneration.from_pretrained('ankur310794/bart-base-keyphrase-generation-kpTimes').to(device)
model.eval()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--version", type=str, default='kp_6.0')
    args = parser.parse_args()
    return args


def get_kps(psgs):
    inputs = tokenizer(psgs, return_tensors='pt', truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, min_length=MIN_LENGTH, max_length=MAX_LENGTH, do_sample=DO_SAMPLE)
    kps = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
    return kps


def cut_psg_word_limit(txt, num_folds, num_words_per_fold):
    """
    Args:
        txt:
        num_folds:
        num_words_per_fold:
    Returns:
        psgs: A list of passage sections
    This function split a passage into shorter sections, each section is then input the keyphrase generation model to
    generate keyphrases
    """
    sents = sent_tokenize(' '.join(txt.split(' ')))
    paras = []
    para, num_words = '', 0
    for sent in sents:
        num_words_sent = len(sent.split(' '))
        para += sent
        num_words += num_words_sent
        if num_words >= num_words_per_fold:
            paras.append(para)
            psg, num_words = '', 0
        if len(paras) >= num_folds:
            break
    # if para != '':
    #     paras.append(para)
    return paras


def combine_kps(kps_batch, urls_batch):
    url2kps_batch = defaultdict(list)
    for kp_str, url in zip(kps_batch, urls_batch):
        url2kps_batch[url] += kp_str.split(';')
    url2kps_batch_ = {}
    for url, kps in url2kps_batch.items():
        url2kps_batch_[url] = list(set(kps))
    return url2kps_batch_


def extract_kps_title(args):
    for split in ['dev', 'train']:
        if not os.path.exists(base_path + '%s/%s-url2titlekps' % (args.version, split)):
            os.mkdir(base_path + '%s/%s-url2titlekps' % (args.version, split))
        print('Loading %s set' % split)
        with open(data_path_in % split) as f:
            lines = f.readlines()
        url2kps = defaultdict(str)
        progress = tqdm(range(len(lines)))
        titles_batch, urls_batch = [], []
        for i, line in enumerate(lines):
            _ = progress.update(1)
            if args.device_idx >= 0:
                if i % 8 != args.device_idx:
                    continue
            info = json.loads(line)
            url = info['url']
            title = info['title']
            titles_batch.append(title)
            urls_batch.append(url)
            if len(titles_batch) >= args.args.batch_size:
                kps_batch = get_kps(titles_batch)
                for url, kps in zip(urls_batch, kps_batch):
                    url2kps[url] = kps
                titles_batch, urls_batch = [], []
        if len(titles_batch) > 0:
            kps_batch = get_kps(titles_batch)
            for url, kps in zip(urls_batch, kps_batch):
                url2kps[url] = kps
        title_kp_path = base_path + '%s/%s-url2titlekps/%s-url2titlekps-%s.json'
        with open(title_kp_path % (args.version, split, split, args.device_idx), 'w') as f:
            json.dump(url2kps, f)


def extract_kps_text(args):
    for split in ['dev']:  # 'train
        if not os.path.exists(base_path + '%s/%s-url2textkps' % (args.version, split)):
            os.mkdir(base_path + '%s/%s-url2textkps' % (args.version, split))
        print('Loading %s set' % split)
        with open(data_path_in % split) as f:
            lines = f.readlines()
        url2kps = defaultdict(str)
        # with open('../recsum_/data/newsroom/%s-url2kps.json' % split) as f:
        #     url2kps = json.load(f)
        progress = tqdm(range(len(lines)))
        paras_batch, urls_batch = [], []
        for i, line in enumerate(lines):
            _ = progress.update(1)
            if args.device_idx >= 0:
                if i % 8 != args.device_idx:
                    continue
            info = json.loads(line)
            if info['text'] == '':
                continue
            url = info['url']
            paras = cut_psg_word_limit(info['text'], NUM_FOLDS, NUM_WORDS_PER_FOLD)
            paras_batch += paras
            urls_batch += [url for _ in paras]
            if len(paras_batch) >= args.batch_size:
                kps_batch = get_kps(paras_batch)
                url2kps_batch = combine_kps(kps_batch, urls_batch)
                url2kps.update(url2kps_batch)
                paras_batch, urls_batch = [], []
        if len(paras_batch) > 0:
            kps_batch = get_kps(paras_batch)
            url2kps_batch = combine_kps(kps_batch, urls_batch)
            url2kps.update(url2kps_batch)
        text_kp_path = base_path + '%s/%s-url2textkps/%s-url2textkps-%s.json'
        with open(text_kp_path % (args.version, split, split, args.device_idx), 'w') as f:
            json.dump(url2kps, f)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(base_path + args.version):
        os.mkdir(base_path + args.version)
    extract_kps_text(args)






