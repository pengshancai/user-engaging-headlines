import os
import json
import torch
from models.selector import load_selector
from models.recommender import load_recommender
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
# from utils.eval_utils import *
import argparse
import pickle
from utils.factcc_utils import evaluate, build_factcc_data
from sentence_transformers import SentenceTransformer
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

BASE_PATH = '~/workspace/'
MAX_LEN_RECOMMENDER = 24


def parse_args():
    parser = argparse.ArgumentParser(description="Generate headlines")
    parser.add_argument("--eval_kp_headline_relevance", action="store_true")
    parser.add_argument("--eval_headline_content_relevance", action="store_true")
    parser.add_argument("--eval_recommendation_scores", action="store_true")
    parser.add_argument("--eval_factcc_scores", action="store_true")
    parser.add_argument("--dataset_file", type=str, help='the dev.jsonl or test.jsonl', default='')
    parser.add_argument("--results_file", type=str, help="the file containing the generated headlines", default='')
    parser.add_argument("--output_file_id", type=str, help="A specific id to mark the evaluation scores' file", default='')
    parser.add_argument("--output_path", type=str, help="the folder to store the evaluation scores' file", default='')
    args = parser.parse_args()
    return args


# ----------------------------------------------------------------------------------------------------------------------
def load_dataset(args):
    with open(args.dataset_file) as f:
        dataset = json.load(f)['data'][:args.num_test]
    return dataset


def load_results(args):
    with open(args.results_file) as f:
        results = json.load(f)
    return results


def load_dpr_models():
    with open(BASE_PATH + 'recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
        args_sl = pickle.load(f)
    encoder_ctx, encoder_query, tokenizer_dpr = load_selector(args_sl)
    encoder_query.eval()
    encoder_ctx.eval()
    encoder_ctx.to(device)
    encoder_query.to(device)
    return encoder_query, encoder_ctx, tokenizer_dpr


def load_recommender_model():
    with open('../recsum_/dump/nr-ft-1.2/args.pkl', 'rb') as f:
        args_rc = pickle.load(f)
        args_rc.recommender_ckpt_path = '/data/home/pengshancai/workspace/PLMNR_/dump/t-1.7.1/epoch-3-30000.pt'
    recommender, tokenizer_rec = load_recommender(args_rc, args_rc.recommender_type)
    recommender.eval()
    recommender.to(device)
    return recommender, tokenizer_rec


def load_bm25_evaluator():
    with open('/data/home/pengshancai/workspace/recsum_/data/newsroom/train-bm25_cache.pkl', 'rb') as f:
        bm25 = pickle.load(f)
    return bm25


def load_factcc_model():
    checkpoint = '/data/home/pengshancai/workspace/recsum_/dump/factcc/factcc-checkpoint/'
    factcc = BertForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    factcc.to(device)
    factcc.eval()
    return factcc, tokenizer


def load_sbert_model():
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    return sbert


# ----------------------------------------------------------------------------------------------------------------------
def get_ut_rel_score_dpr(kp_user, title, encoder_query, encoder_ctx, tokenizer_dpr, device):
    inputs_kp = tokenizer_dpr([kp_user], return_tensors='pt', max_length=32).to(device)
    inputs_title = tokenizer_dpr([title], return_tensors='pt', max_length=128).to(device)
    with torch.no_grad():
        emb_kp = encoder_query(**inputs_kp).pooler_output
        emb_title = encoder_ctx(**inputs_title).pooler_output
        score = torch.matmul(emb_kp, emb_title.T)
    return score.cpu().item()


def get_tt_rel_score_dpr(title, text, encoder_query, encoder_ctx, tokenizer_dpr, device):
    inputs_title = tokenizer_dpr([title], return_tensors='pt', max_length=128).to(device)
    inputs_text = tokenizer_dpr([text], return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        emb_title = encoder_query(**inputs_title).pooler_output
        emb_text = encoder_ctx(**inputs_text).pooler_output
        score = torch.matmul(emb_text, emb_title.T)
    return score.cpu().item()


def get_user_emb(user_histories, recommender, tokenizer_rec, device):
    histories_encoded = tokenizer_rec(user_histories, return_tensors="pt", padding=True, truncation=True,
                                      max_length=MAX_LEN_RECOMMENDER)
    histories_ids, histories_mask = histories_encoded['input_ids'].to(device), histories_encoded['attention_mask'].to(
        device)
    with torch.no_grad():
        input_ids = torch.cat((histories_ids, histories_mask), dim=1)  # Required format of the recommendation model
        log_mask = torch.ones(input_ids.shape[0], dtype=int)
        user_features = torch.unsqueeze(input_ids.to(device), 0)
        log_mask = torch.unsqueeze(log_mask.to(device), 0)
        user_emb = recommender.get_user_emb(user_features, log_mask)
    return user_emb


def get_title_emb(title, recommender, tokenizer_rec, device):
    titles_encoded = tokenizer_rec([title], return_tensors="pt", padding=True, truncation=True,
                                   max_length=MAX_LEN_RECOMMENDER)
    titles_ids, titles_mask = titles_encoded['input_ids'].to(device), titles_encoded['attention_mask'].to(device)
    with torch.no_grad():
        input_ids = torch.cat((titles_ids, titles_mask), dim=1)  # Required format of the recommendation model
        titles_embs = recommender.get_news_emb(torch.unsqueeze(input_ids, dim=0)).squeeze(0)
    return titles_embs


def get_rcmd_score(history, title, recommender, tokenizer_rec, device):
    user_emb = get_user_emb(history, recommender, tokenizer_rec, device).cpu().numpy()[0]
    title_emb = get_title_emb(title, recommender, tokenizer_rec, device).cpu().numpy()[0]
    score = np.dot(user_emb, title_emb)
    return score


# ----------------------------------------------------------------------------------------------------------------------
def eval_kp_headline_relevance_dpr(args, dataset, results):
    encoder_query, encoder_ctx, tokenizer_dpr = load_dpr_models()
    ut_rel_scores_dpr = []
    progress = tqdm(range(len(dataset)))
    for idx, rec in enumerate(dataset):
        _ = progress.update(1)
        src, tgt, pred = results[idx]
        kp_user = rec['kps_user']
        ut_rel_scores_dpr.append(get_ut_rel_score_dpr(kp_user, pred, encoder_query, encoder_ctx, tokenizer_dpr, device))
    with open(args.output_path + args.output_file_id + 'kp_headline_rel.json', 'w') as f:
        pickle.dump(ut_rel_scores_dpr, f)
    print('headline_relevance_dpr_score:\t%s' % np.mean(ut_rel_scores_dpr))


def eval_headline_src_relevance_dpr(args, dataset, results):
    encoder_query, encoder_ctx, tokenizer_dpr = load_dpr_models()
    tt_rel_scores_dpr = []
    progress = tqdm(range(len(dataset)))
    for idx, rec in enumerate(dataset):
        _ = progress.update(1)
        src, tgt, pred = results[idx]
        tt_rel_scores_dpr.append(get_tt_rel_score_dpr(pred, src, encoder_query, encoder_ctx, tokenizer_dpr, device))
    with open(args.output_path + args.output_file_id + 'headline_src_rel.json', 'w') as f:
        pickle.dump(tt_rel_scores_dpr, f)
    print('headline_src_relevance_dpr:\t%s' % np.mean(tt_rel_scores_dpr))


def eval_recommendation_scores(args, dataset, results):
    with open(BASE_PATH + 'recsum_/dump/nr-ft-1.2/args.pkl', 'rb') as f:
        args_rc = pickle.load(f)
        args_rc.recommender_ckpt_path = '/data/home/pengshancai/workspace/PLMNR_/dump/t-1.7.1/epoch-3-30000.pt'
    recommender, tokenizer_rec = load_recommender(args_rc, args_rc.recommender_type)
    rcmd_scores = []
    progress = tqdm(range(len(dataset)))
    for idx, rec in enumerate(dataset):
        _ = progress.update(1)
        history = [title for title in rec['history'].split('; ')]
        src, tgt, pred = results[idx]
        rcmd_score = get_rcmd_score(history, pred, recommender, tokenizer_rec, device)
        rcmd_scores.append(rcmd_score)
    with open(args.output_path + args.output_file_id + 'recommendation_scores.json', 'w') as f:
        pickle.dump(rcmd_scores, f)
    print('recommendation_scores:\t%s' % np.mean(rcmd_scores))


def eval_factcc_scores(args, results):
    factcc, tokenizer_cc = load_factcc_model()
    factcc_temp_path = args.output_path + 'factcc_temp.json'
    srcs = [result[0] for result in results]
    claims = [result[2] for result in results]
    build_factcc_data(srcs, claims, factcc_temp_path)
    factcc_preds = evaluate(factcc, tokenizer_cc, device, factcc_temp_path)
    with open(args.output_path + args.output_file_id + 'factcc_scores.json', 'w') as f:
        pickle.dump(factcc_preds, f)
    print('factcc_scores:\t%s' % np.mean(factcc_preds))


# TODO: Sentence Bert
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args)
    results = load_results(args)
    if args.eval_kp_headline_relevance:
        eval_kp_headline_relevance_dpr(args, dataset, results)
    if args.eval_headline_content_relevance:
        eval_headline_src_relevance_dpr(args, dataset, results)
    if args.eval_recommendation_scores:
        eval_recommendation_scores(args, dataset, results)
    if args.eval_factcc_scores:
        eval_factcc_scores(args, results)


