import jsonlines
import pickle
import random
import torch
from models.selector import load_selector, Selector
from tqdm import tqdm
import numpy as np
import argparse
import os
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate selector")
    parser.add_argument("--dataset_file", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--output_name", type=str, default='')
    parser.add_argument("--selector_dump_dir", type=str, default='')
    parser.add_argument("--kp_select_method", type=str, default='')
    parser.add_argument("--top_ks", type=str, default='1/3/5')
    args = parser.parse_args()
    return args


def load_test_dataset(args):
    with jsonlines.open(args.dataset_file) as f:
        dataset = [line for line in f]
    return dataset


def load_ckpt(ckpt_dir, partition_key='module.'):
    if 'last.ckpt' not in ckpt_dir:
        ckpt_dir = ckpt_dir + 'lightning_logs/version_0/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_dir)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


def load_selector_naive(args):
    # with open(args.base_dir + 'recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
    with open(args.selector_dump_dir + 'args.pkl', 'rb') as f:
        args_selector = pickle.load(f)
    encoder_query, encoder_ctx, tokenizer_emb = load_selector(args_selector)
    encoder_query.eval()
    encoder_ctx.eval()
    encoder_query.to(device)
    encoder_ctx.to(device)
    return encoder_query, encoder_ctx, tokenizer_emb


def load_selector_late(args):
    # args.selector_dump_dir = "../recsum_/dump/nr-sl-2.1/" | "../recsum_/dump/gw-sl-2.0/"
    with open(args.selector_dump_dir + 'args.pkl', 'rb') as f:
        args_selector = pickle.load(f)
    encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_selector)
    model_sl = Selector(args_selector, encoder_src, encoder_tgt, tokenizer_emb)
    ckpt_sl = load_ckpt(args.selector_dump_dir)
    model_sl.load_state_dict(ckpt_sl['state_dict'])
    encoder_kp, encoder_user = model_sl.encoder_src.to(device), model_sl.encoder_tgt.to(device)
    encoder_kp.eval()
    encoder_user.eval()
    return encoder_kp, encoder_user, tokenizer_emb


def load_selector_early(args):
    # args.selector_dump_dir = "../recsum_/dump/nr-sl-3.1/" | "../recsum_/dump/gw-sl-3.0/"
    with open(args.selector_dump_dir + 'args.pkl', 'rb') as f:
        args_selector = pickle.load(f)
    encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_selector)
    model_sl = Selector(args_selector, encoder_src, encoder_tgt, tokenizer_emb)
    ckpt_sl = load_ckpt(args.selector_dump_dir)
    model_sl.load_state_dict(ckpt_sl['state_dict'])
    encoder_kp, encoder_user = model_sl.encoder_src.to(device), model_sl.encoder_tgt.to(device)
    encoder_kp.eval()
    encoder_user.eval()
    return encoder_kp, encoder_user, tokenizer_emb


def select_kps_early(kps, user_titles, encoder_query, encoder_ctx, tokenizer_emb, select_method='max', top_k=1):
    if type(kps) == str:
        kps = kps.split(';')
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


def select_kps_late(kps, user, encoder_kp, encoder_user, tokenizer_emb, top_k=1):
    if type(kps) == str:
        kps = kps.split(';')
    inputs_kps = tokenizer_emb(kps, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
    inputs_user = tokenizer_emb([user], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        embs_kps = encoder_kp(**inputs_kps).pooler_output
        embs_user = encoder_user(**inputs_user).pooler_output
    logits = torch.matmul(embs_kps, embs_user.T).reshape(-1)
    ranked_ids = torch.argsort(logits, descending=True).cpu().numpy()
    return [kps[idx] for idx in ranked_ids[: top_k]]


def select_kps_random(kps, top_k=1):
    if type(kps) == str:
        kps = kps.split(';')
    random.shuffle(kps)
    if top_k > 0:
        top_k = min(top_k, len(kps))
        return random.choices(kps, k=top_k)
    else:
        return kps


def get_top_k_hit(kps_focus, kp_target, k):
    if kp_target in kps_focus[:k]:
        return 1
    else:
        return 0


def get_rank(kps_focus, kp_target):
    return kps_focus.index(kp_target)


def update_output_file(args, metric_name, scores):
    output_name = args.output_name
    print('%s@%s:\t%s' % (output_name, metric_name, np.mean(scores)))
    output_file = args.output_dir + output_name + '.pkl'
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            all_scores = pickle.load(f)
    else:
        all_scores = {}
    all_scores[metric_name] = scores
    with open(output_file, 'wb') as f:
        pickle.dump(all_scores, f)
    print('Score saved to %s' % output_file)


if __name__ == '__main__':
    args = parse_args()
    with open('../recsum_/za/args/args_sel.pkl', 'wb') as f:
        pickle.dump(args, f)
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_test_dataset(args)
    assert args.kp_select_method in {'early-ft', 'early-naive', 'late-ft', 'late-naive', 'random'}
    if args.kp_select_method == 'early-ft':
        encoder_kp, encoder_user, tokenizer_emb = load_selector_early(args)
    elif args.kp_select_method == 'late-ft':
        encoder_kp, encoder_user, tokenizer_emb = load_selector_late(args)
    elif args.kp_select_method in {'early-naive', 'late-naive'}:
        encoder_kp, encoder_user, tokenizer_emb = load_selector_naive(args)
    else:
        encoder_kp, encoder_user, tokenizer_emb = None, None, None
    top_k_hits = {int(top_k):[] for top_k in args.top_ks.split('/')}
    ranks = []
    progress = tqdm(range(len(dataset)))
    for idx, rec in enumerate(dataset):
        if args.kp_select_method.startswith('early'):
            user_titles = rec['history'].split('; ')
            kps_focus = select_kps_early(rec['kps_doc'], user_titles, encoder_kp, encoder_user, tokenizer_emb, 'max', 100)
        elif args.kp_select_method.startswith('late'):
            kps_focus = select_kps_late(rec['kps_doc'], rec['history'], encoder_kp, encoder_user, tokenizer_emb, 100)
        elif args.kp_select_method.startswith('random'):
            kps_focus = select_kps_random(rec['kps_doc'], top_k=-1)
        kp_target = rec['kp_user_selected']
        for k in top_k_hits.keys():
            top_k_hit = get_top_k_hit(kps_focus, kp_target, k)
            top_k_hits[k].append(top_k_hit)
        ranks.append(get_rank(kps_focus, kp_target))
        _ = progress.update(1)
    for k, values in top_k_hits.items():
        update_output_file(args, 'selector_top_%s_hit' % k, values)
    update_output_file(args, 'ranks', ranks)















# with open('../recsum_/za/args/args_gen.pkl', 'rb') as f:
#     args = pickle.load(f)
#     args.base_dir = '../'
#     args.generator_dump_dir = "../recsum_/dump/gw-pt-3.0/"
#     args.selector_dump_dir = "../recsum_/dump/gw-sl-2.0/"
#     args.dataset_file = '../recsum_/data/gigaword/synthesized_user/test.json'
#     args.kp_select_method = 'early-ft'
#     args.top_ks = '1/3/5'


