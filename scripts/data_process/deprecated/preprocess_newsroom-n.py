"""
Preprocess the dataset without any modification
"""
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
from models.summarizer import load_summarizer
from models.general import SummarizerPreTrain
from utils.data_utils import DatasetRecSum, DatasetSumm, DatasetUser, DataCollatorForRecSum, DataModuleRecSum
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import load_dataset
from utils.reward_utils import load_retriever, RetrievalScorer, RecommenderScorer
import torch

with open('../recsum_/dump/nr-ft-1.1/args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.source_prefix = ''

summarizer, tokenizer = load_summarizer(args)
model = SummarizerPreTrain(args, summarizer, tokenizer)
data_files_summ = {}
if args.train_file_summ is not None:
    data_files_summ["train"] = args.train_file_summ

if args.validation_file_summ is not None:
    data_files_summ["validation"] = args.validation_file_summ

extension = data_files_summ[list(data_files_summ.keys())[0]].split(".")[-1]
raw_datasets_summ = load_dataset(extension, data_files=data_files_summ, field='data')
datasets_summ = {}
for split in raw_datasets_summ:
    datasets_summ[split] = DatasetSumm(args, raw_datasets_summ[split], tokenizer, 'cache-%s' % split)

dataset_user = DatasetUser(args.data_file_user)
train_dataset = DatasetRecSum(datasets_summ['train'], dataset_user)
valid_dataset = DatasetRecSum(datasets_summ['validation'], dataset_user)

label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
args.label_pad_token_id = label_pad_token_id
data_collator = DataCollatorForRecSum(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=None,
    padding=True,
)
data_module = DataModuleRecSum(args, data_collator, train_dataset, valid_dataset)
