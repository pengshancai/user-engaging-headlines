import json
import jsonlines
import os
import pickle
import random
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from models.summarizer import load_summarizer
from models.selector import load_selector, Selector
from models.general import SummarizerPreTrainNaive

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Generate headlines")
    parser.add_argument("--generator_dump_dir", type=str, default='')
    parser.add_argument("--selector_dump_dir", type=str, default='')
    parser.add_argument("--dataset_file", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=32)
    parser.add_argument("--kp_select_method", type=str, default='')
    parser.add_argument("--top_k", type=int, default=-1)
    # parser.add_argument("--num_test", type=int, default=10000)
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


def load_summarizer_naive(args):
    # args.generator_dump_dir = "../recsum_/dump/gw-pt-3.0/" | "../recsum_/dump/nr-pt-3.0/"
    with open(args.generator_dump_dir + 'args.pkl', 'rb') as f:
        args_generator = pickle.load(f)
    if 'nr-pt' in args.generator_dump_dir:
        summarizer, tokenizer_sum = load_summarizer(args_generator, added_tokens=['<u>'])
    else:
        summarizer, tokenizer_sum = load_summarizer(args_generator)
    model_base = SummarizerPreTrainNaive(args_generator, summarizer, tokenizer_sum)
    ckpt_base = load_ckpt(args.generator_dump_dir)
    model_base.load_state_dict(ckpt_base['state_dict'])
    summarizer_base = model_base.summarizer
    summarizer_base.to(device)
    return summarizer_base, tokenizer_sum


def load_summarizer_prompt(args):
    # args.generator_dump_dir = "../recsum_/dump/gw-pt-3.1/" | "../recsum_/dump/nr-pt-3.1/"
    with open(args.generator_dump_dir + 'args.pkl', 'rb') as f:
        args_generator = pickle.load(f)
    summarizer, tokenizer_sum = load_summarizer(args_generator)
    model_prompt = SummarizerPreTrainNaive(args_generator, summarizer, tokenizer_sum)
    ckpt_prompt = load_ckpt(args.generator_dump_dir)
    model_prompt.load_state_dict(ckpt_prompt['state_dict'])
    summarizer_prompt = model_prompt.summarizer
    summarizer_prompt.to(device)
    summarizer_prompt.eval()
    return summarizer_prompt, tokenizer_sum


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
    # args.selector_dump_dir  = "../recsum_/dump/nr-sl-2.1/" | "../recsum_/dump/gw-sl-2.0/"
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
    top_k = min(top_k, len(kps))
    return random.choices(kps, k=top_k)


def prepare_inputs(args, rec, encoder_kp, encoder_user, tokenizer_sum, tokenizer_emb, top_k, kp_select_method):
    if kp_select_method == 'none-kp':
        src_ = rec['src']
    elif kp_select_method == 'gold-kp':
        kps_gold = rec['kps_title'].split(';')
        if top_k > 0:
            kps_gold = kps_gold[: top_k]
        kps_focus = '; '.join(kps_gold)
        src_ = kps_focus + '</s> ' + rec['src']
    elif kp_select_method in {'early-ft', 'early-naive'}:
        user_titles = rec['history'].split('; ')
        kps_focus = "; ".join(select_kps_early(rec['kps_doc'], user_titles, encoder_kp, encoder_user, tokenizer_emb, 'max', top_k))
        src_ = kps_focus + '</s> ' + rec['src']
    elif kp_select_method in {'late-ft', 'late-naive'}:
        kps_focus = "; ".join(select_kps_late(rec['kps_doc'], rec['history'], encoder_kp, encoder_user, tokenizer_emb, top_k))
        src_ = kps_focus + '</s> ' + rec['src']
    else:
        assert kp_select_method == 'random'
        kps_focus = "; ".join(select_kps_random(rec['kps_doc'], top_k))
        src_ = kps_focus + '</s> ' + rec['src']
    inputs = tokenizer_sum(src_, return_tensors='pt', max_length=args.max_src_len, truncation=True).to(device)
    if args.max_tgt_len:
        inputs['max_length'] = args.max_tgt_len
    return inputs


def generate_results(args):
    dataset = load_test_dataset(args)
    assert args.kp_select_method in {'none-kp', 'gold-kp', 'early-ft', 'early-naive', 'late-ft', 'late-naive', 'random'}
    if args.kp_select_method == 'none-kp':
        summarizer, tokenizer_sum = load_summarizer_naive(args)
        encoder_kp, encoder_user, tokenizer_emb = None, None, None
    else:
        summarizer, tokenizer_sum = load_summarizer_prompt(args)
        if args.kp_select_method == 'early-ft':
            encoder_kp, encoder_user, tokenizer_emb = load_selector_early(args)
        elif args.kp_select_method == 'late-ft':
            encoder_kp, encoder_user, tokenizer_emb = load_selector_late(args)
        elif args.kp_select_method in {'early-naive', 'late-naive'}:
            encoder_kp, encoder_user, tokenizer_emb = load_selector_naive(args)
        else:
            encoder_kp, encoder_user, tokenizer_emb = None, None, None
    progress = tqdm(range(len(dataset)))
    preds_all = []
    for idx, rec in enumerate(dataset):
        _ = progress.update(1)
        inputs = prepare_inputs(args, rec, encoder_kp, encoder_user, tokenizer_sum, tokenizer_emb, args.top_k, args.kp_select_method)
        outputs = summarizer.generate(**inputs).cpu()
        pred = tokenizer_sum.decode(outputs.numpy()[0], skip_special_tokens=True)
        preds_all.append((rec['src'], rec['tgt'], pred))
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True)
    if not args.output_file.endswith('.json'):
        args.output_file = args.output_fileload_selector + '.json'
    with open(args.output_dir + args.output_file, 'w') as f:
        json.dump(preds_all, f)


if __name__ == "__main__":
    args = parse_args()
    # with open('../recsum_/za/args/args_gen.pkl', 'wb') as f:
    #     pickle.dump(args, f)
    # exit()
    generate_results(args)
