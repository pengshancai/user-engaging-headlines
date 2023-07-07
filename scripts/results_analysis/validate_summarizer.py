import json
import numpy as np
import torch
from models.summarizer import load_summarizer
from utils.data_utils import DatasetSumm, DatasetUser, DataCollatorForRecSum, DataModuleRecSum, DatasetRecSum
from datasets import load_dataset
import pickle
from torch.utils.data import DataLoader
from models.general import SummarizerPreTrain
from tqdm import tqdm
from rouge_score import rouge_scorer
from collections import defaultdict
import os

"""
This script is to evaluate and single out the best performance checkpoint
"""

with open('../recsum_/dump/nr-pt-1.3/args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.data_file_user = '../recsum_/data/processed/world_user_emb.pkl'
    args.per_device_eval_batch_size = 32

"""
Step 1: Load the model
"""


def load_checkpoint(checkpoint_path, partition_key='module.'):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    checkpoint['state_dict'] = state_dict
    return checkpoint


summarizer, tokenizer = load_summarizer(args)
model = SummarizerPreTrain(args, summarizer, tokenizer)
device = torch.device('cuda')
model.to(device)

"""
Step 2: Load dataset
"""
data_files_summ = {}
if args.validation_file_summ is not None:
    data_files_summ["validation"] = args.validation_file_summ

extension = data_files_summ[list(data_files_summ.keys())[0]].split(".")[-1]
raw_datasets_summ = load_dataset(extension, data_files=data_files_summ, field='data')
datasets_summ = {}
for split in raw_datasets_summ:
    datasets_summ[split] = DatasetSumm(args, raw_datasets_summ[split], tokenizer, 'cache-%s' % split)

dataset_user = DatasetUser(args.data_file_user)
valid_dataset = DatasetRecSum(datasets_summ['validation'], dataset_user)
data_collator = DataCollatorForRecSum(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    pad_to_multiple_of=None,
    padding=True,
)

valid_dataloader = DataLoader(
    valid_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=args.per_device_eval_batch_size,
    num_workers=1,
    drop_last=True,
    pin_memory=True,
)


def get_val_results(ckpt_name):
    # Load checkpoint
    ckpt_path = ckpt_base_path + ckpt_name + '/checkpoint/mp_rank_00_model_states.pt'
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    # Do prediction
    preds_all, golds_all = [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    progress = tqdm(range(NUM_VAL_STEPS))
    for i, batch in enumerate(valid_dataloader):
        batch = batch.to(device)
        input_ids, attention_mask, labels, decoder_input_ids, user_embs = \
            batch['input_ids'], batch['attention_mask'], batch['labels'], batch['decoder_input_ids'], batch['user_emb']
        encoder_outputs_comb = model.summarizer.get_encoder_output_user_feature(input_ids, attention_mask, user_embs)
        gen_kwargs_sample = {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
            "encoder_outputs": encoder_outputs_comb,
            "do_sample": True,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        gen_results_sample = summarizer.generate(**gen_kwargs_sample)
        output_ids_sample = gen_results_sample['sequences']
        preds = model.summarizer.tokenizer.batch_decode(output_ids_sample, skip_special_tokens=True)
        golds = model.summarizer.tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)
        preds_all += preds
        golds_all += golds
        _ = progress.update(1)
        if i > NUM_VAL_STEPS:
            break
    rouge_scores = defaultdict(list)
    for pred, gold in zip(preds_all, golds_all):
        scores = scorer.score(gold, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    results = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
    }
    print(results)
    # Write results
    with open(ckpt_base_path + ckpt_name + '/results.json', 'w') as f:
        json.dump(results, f)


ckpt_base_path = '/data/home/pengshancai/workspace/recsum_/dump/gw-pt-1.3/lightning_logs/version_0/checkpoints/'
# ckpt_base_path = '/data/home/pengshancai/workspace/recsum_/dump/nr-pt-1.3/lightning_logs/version_0/checkpoints/'
ckpt_names = os.listdir(ckpt_base_path)
ckpt_names.sort()
NUM_VAL_STEPS = 500
for ckpt_name in ckpt_names:
    print('Eval %s' % ckpt_name)
    get_val_results(ckpt_name)

for ckpt_name in ckpt_names:
    with open(ckpt_base_path + ckpt_name + '/results.json') as f:
        results = json.load(f)
    print(ckpt_name + str(results))

# nr: epoch=2-step=18137-valid_loss=1.43461967.ckpt     {'rouge1': 0.3153252222659765, 'rouge2': 0.13804711402694356, 'rougeL': 0.28895041376130276}
# gw: epoch=1-step=80253-valid_loss=1.14992225.ckpt     {'rouge1': 0.36640744436775435, 'rouge2': 0.16231305682590094, 'rougeL': 0.33735714849898707}
