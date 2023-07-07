import json
import pickle
import random

import torch
from models.summarizer import load_summarizer_naive
from models.general import SummarizerPreTrainNaive
from models.selector import load_selector, Selector
from tqdm import tqdm

"""
Step 0: Preparation
"""
path_summarizer = '/cephfs/data/huggingface_models/facebook/bart-large'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_ckpt(ckpt_path, partition_key='module.'):
    if '.ckpt' not in ckpt_path:
        ckpt_path = ckpt_path + 'checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


"""
Step 1: Load dataset
"""
NUM_TEST = 10000
version = '1.3.3'
with open('../recsum_/data/newsroom/dev-kp-history_%s.json' % version) as f:
    dataset = json.load(f)['data'][:NUM_TEST]

"""
Step 2: Load models
"""
with open('../recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
    args_sl = pickle.load(f)

encoder_user, encoder_kp, tokenizer_emb = load_selector(args_sl)
model_sl = Selector(args_sl, encoder_user, encoder_kp, tokenizer_emb)
ckpt_sl_path = '../recsum_/dump/nr-sl-2.0/lightning_logs/version_0/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
ckpt_sl = load_ckpt(ckpt_sl_path)
model_sl.load_state_dict(ckpt_sl['state_dict'])
encoder_user, encoder_kp = model_sl.encoder_user.to(device), model_sl.encoder_kp.to(device)
encoder_ctx, encoder_query, _ = load_selector(args_sl)
encoder_query.to(device)
encoder_ctx.to(device)
encoder_user.eval()
encoder_kp.eval()
encoder_query.eval()
encoder_ctx.eval()

for idx, (n1, p1) in enumerate(encoder_ctx.named_parameters()):
    if idx == 30:
        break

for idx, (n2, p2) in enumerate(encoder_user.named_parameters()):
    if idx == 30:
        break


def select_kps(kps, user, encoder_kp, encoder_user, tokenizer_emb, top_k=1):
    inputs_kps = tokenizer_emb(kps, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    inputs_user = tokenizer_emb([user], return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        embs_kps = encoder_kp(**inputs_kps).pooler_output
        embs_user = encoder_user(**inputs_user).pooler_output
    logits = torch.matmul(embs_kps, embs_user.T).reshape(-1)
    ranked_ids = torch.argsort(logits, descending=True).cpu().numpy()
    return [kps[idx] for idx in ranked_ids[: top_k]]


def select_kps_early(kps, user_titles, encoder_query, encoder_ctx, tokenizer_emb, select_method='max', top_k=1):
    assert select_method in {'max', 'avg'}
    inputs_kps = tokenizer_emb(kps, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    inputs_titles = tokenizer_emb(user_titles, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
        device)
    with torch.no_grad():
        embs_kps = encoder_query(**inputs_kps).pooler_output
        embs_titles = encoder_ctx(**inputs_titles).pooler_output
    logits = torch.matmul(embs_kps, embs_titles.T)
    if select_method == 'max':
        scores, _ = torch.max(logits, dim=1)
    elif select_method == 'avg':
        scores = torch.mean(logits, dim=1)
    else:
        assert False
    ranked_ids = torch.argsort(scores, descending=True).cpu().numpy()
    return [kps[idx] for idx in ranked_ids[: top_k]]


for idx in range(100):
    rec = dataset[idx]
    src = rec['src']
    tgt = rec['tgt']
    user = rec['history']
    kps = rec['kps_doc']
    core_kp = rec['kps_user']
    kps_focus_late = "; ".join(select_kps(kps, user, encoder_kp, encoder_user, tokenizer_emb, 10))
    kps_focus_early_avg = "; ".join(select_kps_early(kps, user, encoder_query, encoder_ctx, tokenizer_emb, 'avg', 10))
    if kps_focus_early_avg != kps_focus_late:
        break

print('core KP:\t%s' % core_kp[0])
print('KPs late:\t%s' % kps_focus_late)
print('KPs early:\t%s' % kps_focus_early_avg)




