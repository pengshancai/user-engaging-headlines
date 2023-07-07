import json
import pickle
import random
import torch
from models.selector import load_selector, Selector
from tqdm import tqdm
import numpy as np

"""
Step 0: Preparation
"""
path_summarizer = '/cephfs/data/huggingface_models/facebook/bart-large'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_ckpt(ckpt_path, partition_key='module.'):
    if 'last.ckpt' not in ckpt_path:
        ckpt_path = ckpt_path + 'checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


NUM_TEST = 10000
kp_version = '7.0'
version = '1.3.3'

with open('../recsum_/data/newsroom/kp_%s/dev-kp-history_%s.json' % (kp_version, version)) as f:
    dataset = json.load(f)['data'][:NUM_TEST]


def get_top_k_hit(kps_focus, kp_target, k):
    if kp_target in kps_focus[:k]:
        return 1
    else:
        return 0


"""
late meet
"""
with open('../recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
    args_sl = pickle.load(f)

encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_sl)
model_sl = Selector(args_sl, encoder_src, encoder_tgt, tokenizer_emb)
ckpt_sl_path = '../recsum_/dump/nr-sl-2.0/lightning_logs/version_3/'
ckpt_sl = load_ckpt(ckpt_sl_path)
model_sl.load_state_dict(ckpt_sl['state_dict'])
encoder_kp, encoder_user = model_sl.encoder_src.to(device), model_sl.encoder_tgt.to(device)

encoder_kp.eval()
encoder_user.eval()


def select_kps_late(kps, user, encoder_kp, encoder_user, tokenizer_emb, top_k=1):
    inputs_kps = tokenizer_emb(kps, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    inputs_user = tokenizer_emb([user], return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        embs_kps = encoder_kp(**inputs_kps).pooler_output
        embs_user = encoder_user(**inputs_user).pooler_output
    logits = torch.matmul(embs_kps, embs_user.T).reshape(-1)
    ranked_ids = torch.argsort(logits, descending=True).cpu().numpy()
    return [kps[idx] for idx in ranked_ids[: top_k]]


progress = tqdm(range(len(dataset)))
top1_hits, top3_hits, top5_hits = [], [], []
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    kps = rec['kps_doc']
    user = rec['history']
    kps_focus = select_kps_late(kps, user, encoder_kp, encoder_user, tokenizer_emb, 5)
    kp_target = rec['kps_user']
    if type(kp_target) == list:
        kp_target = kp_target[0]
    top1_hit = get_top_k_hit(kps_focus, kp_target, 1)
    top3_hit = get_top_k_hit(kps_focus, kp_target, 3)
    top5_hit = get_top_k_hit(kps_focus, kp_target, 5)
    top1_hits.append(top1_hit)
    top3_hits.append(top3_hit)
    top5_hits.append(top5_hit)
    if idx % 1000 == 0 and idx > 0:
        print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
        print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
        print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))

print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))


# ----------------------------------------------------------------------------------------------------------------------
"""
Early meet
"""

with open('../recsum_/dump/nr-sl-3.0/args.pkl', 'rb') as f:
    args_sl = pickle.load(f)

encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_sl)
model_sl = Selector(args_sl, encoder_src, encoder_tgt, tokenizer_emb)
ckpt_sl_path = '../recsum_/dump/nr-sl-3.0/lightning_logs/version_2/'
ckpt_sl = load_ckpt(ckpt_sl_path)
model_sl.load_state_dict(ckpt_sl['state_dict'])
encoder_kp, encoder_title = model_sl.encoder_src.to(device), model_sl.encoder_tgt.to(device)
encoder_kp.eval()
encoder_title.eval()


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


progress = tqdm(range(len(dataset)))
top1_hits, top3_hits, top5_hits = [], [], []
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    kps = rec['kps_doc']
    user_titles = rec['history'].split('; ')
    kps_focus = select_kps_early(kps, user_titles, encoder_kp, encoder_title, tokenizer_emb, 'max', 5)
    kp_target = rec['kps_user']
    if type(kp_target) == list:
        kp_target = kp_target[0]
    top1_hit = get_top_k_hit(kps_focus, kp_target, 1)
    top3_hit = get_top_k_hit(kps_focus, kp_target, 3)
    top5_hit = get_top_k_hit(kps_focus, kp_target, 5)
    top1_hits.append(top1_hit)
    top3_hits.append(top3_hit)
    top5_hits.append(top5_hit)
    if idx % 1000 == 0 and idx > 0:
        print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
        print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
        print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))

print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))


"""
Naive
"""
encoder_query, encoder_ctx, tokenizer_emb = load_selector(args_sl)
encoder_query.eval()
encoder_ctx.eval()
encoder_query.to(device)
encoder_ctx.to(device)


def select_kps_naive(kps, history, encoder_query, encoder_ctx, tokenizer_emb, top_k=1):
    inputs_kps = tokenizer_emb(kps, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    inputs_history = tokenizer_emb([history], return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        embs_kps = encoder_query(**inputs_kps).pooler_output
        embs_user = encoder_ctx(**inputs_history).pooler_output
    logits = torch.matmul(embs_kps, embs_user.T).reshape(-1)
    ranked_ids = torch.argsort(logits, descending=True).cpu().numpy()
    return [kps[idx] for idx in ranked_ids[: top_k]]


progress = tqdm(range(len(dataset)))
top1_hits, top3_hits, top5_hits = [], [], []
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    src = rec['src']
    tgt = rec['tgt']
    kps = rec['kps_doc']
    user = rec['history']
    kps_focus = select_kps_naive(kps, user, encoder_query, encoder_ctx, tokenizer_emb, 5)
    kp_target = rec['kps_user']
    if type(kp_target) == list:
        kp_target = kp_target[0]
    top1_hit = get_top_k_hit(kps_focus, kp_target, 1)
    top3_hit = get_top_k_hit(kps_focus, kp_target, 3)
    top5_hit = get_top_k_hit(kps_focus, kp_target, 5)
    top1_hits.append(top1_hit)
    top3_hits.append(top3_hit)
    top5_hits.append(top5_hit)
    if idx % 1000 == 0 and idx > 0:
        print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
        print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
        print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))

print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))


"""
Random
"""


def select_kps_random(kps, top_k=1):
    top_k = min(top_k, len(kps))
    return random.choices(kps, k=top_k)


progress = tqdm(range(len(dataset)))
top1_hits, top3_hits, top5_hits = [], [], []
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    src = rec['src']
    tgt = rec['tgt']
    kps = rec['kps_doc']
    user = rec['history']
    kps_focus = select_kps_random(kps, 5)
    kp_target = rec['kps_user']
    if type(kp_target) == list:
        kp_target = kp_target[0]
    top1_hit = get_top_k_hit(kps_focus, kp_target, 1)
    top3_hit = get_top_k_hit(kps_focus, kp_target, 3)
    top5_hit = get_top_k_hit(kps_focus, kp_target, 5)
    top1_hits.append(top1_hit)
    top3_hits.append(top3_hit)
    top5_hits.append(top5_hit)
    if idx % 1000 == 0 and idx > 0:
        print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
        print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
        print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))

print('Top 1 Hit:\t%s' % (np.sum(top1_hits) / len(top1_hits)))
print('Top 3 Hit:\t%s' % (np.sum(top3_hits) / len(top3_hits)))
print('Top 5 Hit:\t%s' % (np.sum(top5_hits) / len(top5_hits)))



