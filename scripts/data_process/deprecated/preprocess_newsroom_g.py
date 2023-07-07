"""
Preprocess the dataset by adding noisy key phrase prompt
G: Prompt is sampled from the KPs (just sample 1)
"""
import json
import pickle
import random

from models.summarizer import load_summarizer
from models.general import SummarizerPreTrain
from utils.data_utils import DatasetSumm, DatasetUser, DataCollatorForRecSum, DatasetRecSumPT, DataModuleRecSum
from datasets import load_dataset

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
            kp = random.sample(title_kps, 1)[0]
            recs.append({'text': kp + '</s> ' + text, 'title': title})
    print('Writing files')
    with open('../recsum_/data/newsroom/%s-g.json' % split, 'w') as f:
        info_all = {
            'version': 'G-1.0',
            'data': recs
        }
        json.dump(info_all, f)


with open('../recsum_/dump/nr-pt-3.0/args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.source_prefix = ''
    args.summarizer_model_path = '/cephfs/data/huggingface_models/facebook/bart-base'

# Tokenize the dataset
summarizer, tokenizer = load_summarizer(args)
model = SummarizerPreTrain(args, summarizer, tokenizer)
data_files_summ = {}
data_files_summ["train"] = '../recsum_/data/newsroom/train-g.json'
data_files_summ["validation"] = '../recsum_/data/newsroom/dev-g.json'

extension = data_files_summ[list(data_files_summ.keys())[0]].split(".")[-1]
raw_datasets_summ = load_dataset(extension, data_files=data_files_summ, field='data')
datasets_summ = {}
for split in raw_datasets_summ:
    datasets_summ[split] = DatasetSumm(args, raw_datasets_summ[split], tokenizer, 'cache-%s-g' % split)







