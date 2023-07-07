"""
Preprocess the dataset by adding noisy key phrase prompt
k_1.0: Key phrase prompt generated by the KP generation model
"""
import json
import logging
import argparse
import os
import pickle
import nltk
from filelock import FileLock
from transformers import (
    MODEL_MAPPING,
    SchedulerType,
)
from transformers.utils import is_offline_mode
from models.recommender import load_recommender
from models.summarizer import load_summarizer_naive
from models.general import SummarizerPreTrain
from utils.data_utils import DatasetSumm, DatasetUser, DataCollatorForRecSum, DatasetRecSumPT, DataModuleRecSum
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import load_dataset
from utils.reward_utils import load_retriever, RetrievalScorer, RecommenderScorer
import torch
from tqdm import tqdm

base_path = '../recsum_/data/newsroom/'


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


for split in ['train', 'dev']:
    pass
    with open(base_path + '%s-url2textkps.json' % split) as f:
        url2text_kps = json.load(f)
    with open(base_path + '%s-url2titlekps.json' % split) as f:
        url2title_kps = json.load(f)
    recs = []
    with open(base_path + '%s.jsonl' % split) as json_file:
        for line in json_file:
            info = json.loads(line)
            url = info['url']
            title = process_title(info['title'])
            text = process_text(info['text'])
            title_kps = process_kps(url2title_kps[url])
            text_kps = process_kps(url2text_kps[url])
            recs.append({'text': '; '.join(title_kps) + '</s> ' + text, 'title': title})
    print('Writing files')
    with open('../recsum_/data/newsroom/%s-k_1.0.json' % split, 'w') as f:
        info_all = {
            'version': 'K-1.0',
            'data': recs
        }
        json.dump(info_all, f)


with open('../recsum_/dump/nr-pt-3.0/args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.source_prefix = ''
    args.summarizer_model_path = '/cephfs/data/huggingface_models/facebook/bart-base'

# Tokenize the dataset
summarizer, tokenizer = load_summarizer_naive(args)
model = SummarizerPreTrain(args, summarizer, tokenizer)
data_files_summ = {}
data_files_summ["train"] = '../recsum_/data/newsroom/train-k_1.0.json'
data_files_summ["validation"] = '../recsum_/data/newsroom/dev-k_1.0.json'

extension = data_files_summ[list(data_files_summ.keys())[0]].split(".")[-1]
raw_datasets_summ = load_dataset(extension, data_files=data_files_summ, field='data')
datasets_summ = {}
for split in raw_datasets_summ:
    datasets_summ[split] = DatasetSumm(args, raw_datasets_summ[split], tokenizer, 'cache-%s-k_1.0' % split)








#
pass
with open(base_path + '%s-titles.json' % split) as f:
    titles = json.load(f)