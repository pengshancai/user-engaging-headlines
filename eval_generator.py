import argparse
import json
import os
import pickle
import random
import string
from collections import defaultdict

import numpy as np
import sentence_transformers
import torch
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from rouge_score import rouge_scorer

from models.recommender import load_recommender
from models.selector import load_selector
from utils.eval_utils import get_rcmd_score, Fragments
from utils.factcc_utils import evaluate, build_factcc_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating the generated headlines")
    parser.add_argument("--dataset_file", type=str, default='')
    parser.add_argument("--result_file", type=str, default='')
    parser.add_argument("--result_column_idx", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--output_name", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


def load_test_dataset(args):
    dataset_files = {'test': args.dataset_file}
    dataset = load_dataset('json', data_files=dataset_files)['test']
    return dataset


def load_test_results(args):
    with open(args.result_file) as f:
        results = json.load(f)
    results = [result[args.result_column_idx] for result in results]
    return results


def update_output_file(args, metric_name, scores):
    if args.output_name != '':
        output_name = args.output_name
    else:
        output_name = args.result_file.split('/')[-1].split('.')[0]
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


def get_user_title_relevance_dpr(args, dataset, results, device):
    ut_rel_scores_dpr = []
    with open('../recsum_/dump/nr-sl-2.1/args.pkl', 'rb') as f:
        args_sl = pickle.load(f)
    encoder_ctx, encoder_query, tokenizer_dpr = load_selector(args_sl)
    encoder_ctx.to(device)
    encoder_query.to(device)
    encoder_ctx.eval()
    encoder_query.eval()
    num_batches = int(len(dataset) / args.batch_size)
    progress = tqdm(range(num_batches + 1), desc='calculating user-title relevance DPR')
    for batch_idx in range(num_batches + 1):
        ids = list(range(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, len(dataset))))
        if len(ids) == 0: break
        _ = progress.update(1)
        core_kps = dataset['kp_user_selected'][ids[0]: ids[-1] + 1]
        preds = [results[idx] for idx in ids]
        inputs_title = tokenizer_dpr(preds, return_tensors='pt', max_length=128, padding=True, truncation=True).to(
            device)
        inputs_kps = tokenizer_dpr(core_kps, return_tensors='pt', max_length=32, padding=True, truncation=True).to(
            device)
        with torch.no_grad():
            emb_kps = encoder_query(**inputs_kps).pooler_output
            emb_title = encoder_ctx(**inputs_title).pooler_output
            scores = list(torch.sum(emb_kps * emb_title, dim=1).cpu().numpy())
            ut_rel_scores_dpr.extend(scores)
    update_output_file(args, 'user-title relevance dpr', ut_rel_scores_dpr)


def get_user_title_relevance_sbert(args, dataset, results, device):
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    ut_rel_scores_sbert = []
    num_batches = int(len(dataset) / args.batch_size)
    progress = tqdm(range(num_batches + 1), desc='calculating user-title relevance SBERT')
    for batch_idx in range(num_batches + 1):
        ids = list(range(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, len(dataset))))
        if len(ids) == 0: break
        _ = progress.update(1)
        core_kps = dataset['kp_user_selected'][ids[0]: ids[-1] + 1]
        preds = [results[idx] for idx in ids]
        with torch.no_grad():
            emb_kps = sbert.encode(core_kps, convert_to_tensor=True)
            emb_preds = sbert.encode(preds, convert_to_tensor=True)
        cos_scores = [sentence_transformers.util.cos_sim(emb_kps[i], emb_preds[i]).cpu().item() for i in
                      range(len(ids))]
        ut_rel_scores_sbert.extend(cos_scores)
    update_output_file(args, 'user-title relevance sbert', ut_rel_scores_sbert)


def get_recommendation_scores(args, dataset, results, device):
    with open('../recsum_/dump/mind/args.pkl', 'rb') as f:
        args_rc = pickle.load(f)
        args_rc.recommender_ckpt_path = '../recsum_/dump/mind/epoch-3-30000.pt'
        args_rc.recommender_model_path = 'facebook/bart-base'
    recommender, tokenizer_rec = load_recommender(args_rc, args_rc.recommender_type)
    recommender.to(device)
    recommender.eval()
    rcmd_scores = []
    progress = tqdm(range(len(dataset)), desc='calculating recommendation scores')
    for idx, history in enumerate(dataset['history']):
        headline = results[idx]
        rcmd_score = get_rcmd_score(history, headline, recommender, tokenizer_rec, device)
        rcmd_scores.append(rcmd_score)
        _ = progress.update(1)
    update_output_file(args, 'recommendation scores', rcmd_scores)


def get_title_text_relevance_dpr(args, dataset, results, device):
    tt_rel_scores_dpr = []
    with open('../recsum_/dump/nr-sl-2.1/args.pkl', 'rb') as f:
        args_sl = pickle.load(f)
    encoder_ctx, encoder_query, tokenizer_dpr = load_selector(args_sl)
    encoder_ctx.to(device)
    encoder_query.to(device)
    encoder_ctx.eval()
    encoder_query.eval()
    num_batches = int(len(dataset) / args.batch_size)
    progress = tqdm(range(num_batches + 1), desc='calculating text-title relevance DPR')
    for batch_idx in range(num_batches + 1):
        ids = list(range(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, len(dataset))))
        if len(ids) == 0: break
        _ = progress.update(1)
        srcs = dataset['src'][ids[0]: ids[-1] + 1]
        preds = [results[idx] for idx in ids]
        inputs_title = tokenizer_dpr(preds, return_tensors='pt', max_length=128, padding=True, truncation=True).to(
            device)
        inputs_text = tokenizer_dpr(srcs, return_tensors='pt', max_length=512, padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb_title = encoder_query(**inputs_title).pooler_output
            emb_text = encoder_ctx(**inputs_text).pooler_output
            scores = list(torch.sum(emb_title * emb_text, dim=1).cpu().numpy())
            tt_rel_scores_dpr.extend(scores)
    update_output_file(args, 'title-text relevance_dpr', tt_rel_scores_dpr)


def get_title_text_relevance_sbert(args, dataset, results, device):
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    tt_rel_scores_sbert = []
    num_batches = int(len(dataset) / args.batch_size)
    progress = tqdm(range(num_batches + 1), desc='calculating title-text relevance SBERT')
    for batch_idx in range(num_batches + 1):
        ids = list(range(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, len(dataset))))
        if len(ids) == 0: break
        _ = progress.update(1)
        preds = [results[idx] for idx in ids]
        texts = dataset['src'][ids[0]: ids[-1] + 1]
        sents = []
        pred2sent = {}
        for idx_title, text in enumerate(texts):
            sents_text = sent_tokenize(text)
            pred2sent[idx_title] = list(range(len(sents), len(sents) + len(sents_text)))
            sents.extend(sents_text)
        with torch.no_grad():
            emb_preds = sbert.encode(preds, convert_to_tensor=True)
            emb_sents = sbert.encode(sents, convert_to_tensor=True)
        for idx_title, idx_sents in pred2sent.items():
            cos_scores_text = [
                sentence_transformers.util.cos_sim(emb_preds[idx_title], emb_sents[idx_sent]).cpu().item() for idx_sent
                in idx_sents]
            tt_rel_scores_sbert.append(max(cos_scores_text))
    update_output_file(args, 'title-text relevance sbert', tt_rel_scores_sbert)


def get_title_text_factcc(args, dataset, results, device):
    checkpoint = '../recsum_/dump/factcc/factcc-checkpoint/'
    factcc = BertForSequenceClassification.from_pretrained(checkpoint)
    tokenizer_cc = BertTokenizer.from_pretrained('bert-base-uncased')
    factcc.to(device)
    factcc.eval()
    srcs = dataset['src']
    claims = results
    temp_file_path = args.output_dir + '%s.json' % ''.join(random.sample(string.ascii_letters + string.digits, 15))
    build_factcc_data(srcs, claims, temp_file_path)
    pred_scores = evaluate(factcc, tokenizer_cc, device, temp_file_path)
    update_output_file(args, 'title-text consistency factcc', pred_scores)
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


def get_extractive_features(args, dataset, results, device):
    extractive_features = defaultdict(list)
    progress = tqdm(range(len(results)), desc='calculating extractive features')
    for src, pred in zip(dataset['src'], results):
        frag = Fragments(pred, src)
        extractive_features['lengths'].append(len(pred.split(' ')))
        extractive_features['coverage_scores'].append(frag.coverage())
        extractive_features['density_scores'].append(frag.density())
        extractive_features['compression_scores'].append(frag.compression())
        _ = progress.update(1)
    update_output_file(args, 'length', extractive_features['lengths'])
    update_output_file(args, 'coverage_scores', extractive_features['coverage_scores'])
    update_output_file(args, 'density_scores', extractive_features['density_scores'])
    update_output_file(args, 'compression_scores', extractive_features['compression_scores'])


def get_rouge_score(args, dataset, results, device):
    progress = tqdm(range(len(results)))
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
    scores_all = defaultdict(list)
    for gold, pred in zip(dataset['tgt'], results):
        scores = scorer.score(gold, pred)
        _ = progress.update(1)
        for metric in rouge_metrics:
            scores_all[metric].append(scores[metric].fmeasure)
    for metric in rouge_metrics:
        update_output_file(args, metric, scores_all[metric])


if __name__ == "__main__":
    args = parse_args()
    dataset = load_test_dataset(args)
    results = load_test_results(args)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    get_user_title_relevance_dpr(args, dataset, results, device)
    get_user_title_relevance_sbert(args, dataset, results, device)
    get_recommendation_scores(args, dataset, results, device)
    get_title_text_relevance_dpr(args, dataset, results, device)
    get_title_text_relevance_sbert(args, dataset, results, device)
    get_title_text_factcc(args, dataset, results, device)
    get_extractive_features(args, dataset, results, device)
    get_rouge_score(args, dataset, results, device)


# with open('../recsum_/za/args/args_gen.pkl', 'rb') as f:
#     args = pickle.load(f)
#     args.dataset_file = '../recsum_/data/gigaword/synthesized_user/test.json'
#     args.result_column_idx = 2
#     args.result_file = '../recsum_/results/gigaword/kp-late-ft.json'
#     args.result_file = '../recsum_/results/gigaword/none-kp.json'
#     args.output_dir = '../recsum_/results/gigaword/scores/'
#     args.batch_size = 32
