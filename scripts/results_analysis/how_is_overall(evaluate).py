"""
Deprecated
"""

import os
import json
from models.selector import load_selector
from models.recommender import load_recommender
from collections import defaultdict
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from utils.eval_utils import *
import pickle
from utils.factcc_utils import evaluate, build_factcc_data
from sentence_transformers import SentenceTransformer

"""
Step 0: Preparation
"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"""
Step 1: Load datasets and results
"""
kp_version = '1.0'
version = '1.3.1'
NUM_TEST = 10000
with open('../recsum_/data/newsroom/kp_%s/dev-kp-history_%s.json' % (kp_version, version)) as f:
    dataset = json.load(f)['data'][:NUM_TEST]


gen_results = {}

with open('../recsum_/results/newsroom/kp_%s/nr-pt-3.0-%s.json' % (kp_version, version)) as f:
    lines_base = json.load(f)
    gen_results['srcs'] = [rec[0] for rec in lines_base]
    gen_results['originals'] = [rec[1] for rec in lines_base]
    gen_results['none-kp'] = [rec[2] for rec in lines_base]

with open('../recsum_/results/newsroom/kp_%s/nr-gold-%s.json' % (kp_version, version)) as f:
    lines_gold = json.load(f)
    gen_results['gold_kps'] = [rec[2] for rec in lines_gold]

# exp_names = ['late', 'early-max', 'early-avg', 'naive', 'random']
exp_names = ['late', 'early-max', 'naive', 'random']
for exp_name in exp_names:
    for top_k in [1, 2, 3, 4, 5]:
        with open('../recsum_/results/newsroom/kp_%s/nr-sl-%s-2.0-top-%s-%s.json' % (kp_version, exp_name, top_k, version)) as f:
            lines = json.load(f)
        gen_results['%s-%s' % (exp_name, top_k)] = [rec[2] for rec in lines]

# target_exp_names = list(set(gen_results.keys()).difference(set(['srcs'])))

# exp_cates = ['late', 'early-max', 'early-avg', 'naive', 'random']
exp_cates = ['late', 'early-max', 'naive', 'random']
target_exp_names = ['originals', 'none-kp', 'gold_kps'] + \
            [exp_cate + '-1' for exp_cate in exp_cates] + \
            [exp_cate + '-2' for exp_cate in exp_cates] + \
            [exp_cate + '-3' for exp_cate in exp_cates] + \
            [exp_cate + '-4' for exp_cate in exp_cates] + \
            [exp_cate + '-5' for exp_cate in exp_cates]


"""
Step 2: Load models
"""

# gpt2_path = '/data/home/pengshancai/workspace/recsum_/dump/lm-nr'
# tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_path)
# gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_path)
# gpt2.to(device)
# gpt2.eval()

# bert_path = '/data/home/pengshancai/workspace/recsum_/dump/mlm-nr'
# tokenizer_bert = AutoTokenizer.from_pretrained(bert_path)
# bert = AutoModelForMaskedLM.from_pretrained(bert_path)
# bert.to(device)
# bert.eval()

"""
Step 3: Evaluate
"""
with open('../recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
    args_sl = pickle.load(f)

encoder_ctx, encoder_query, tokenizer_dpr = load_selector(args_sl)
encoder_query.eval()
encoder_ctx.eval()
encoder_ctx.to(device)
encoder_query.to(device)

# 3.1.1 User KP - title relevance (DPR)
ut_rel_scores_dpr = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    kp_user = rec['kps_user']
    if type(kp_user) == list:
        kp_user = kp_user[0]
    for key in gen_results.keys():
        if key not in target_exp_names:
            continue
        headline = gen_results[key][idx]
        ut_rel_scores_dpr[key].append(get_ut_rel_score_dpr(kp_user, headline, encoder_query, encoder_ctx, tokenizer_dpr, device))
    if idx % 1000 == 0 and idx > 0:
        for key, values in ut_rel_scores_dpr.items():
            print(key + ':\t' + str(np.mean(values)))

for key, values in ut_rel_scores_dpr.items():
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-ut_rel_scores_dpr-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(ut_rel_scores_dpr, f)

# exit()

# 3.1.2 title-text relevance score (DPR)
tt_rel_scores_dpr = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    src = gen_results['srcs'][idx]
    for key in gen_results.keys():
        if key not in target_exp_names:
            continue
        headline = gen_results[key][idx]
        tt_rel_scores_dpr[key].append(
            get_tt_rel_score_dpr(headline, src, encoder_query, encoder_ctx, tokenizer_dpr, device))
    if idx % 1000 == 0 and idx > 0:
        for key, values in tt_rel_scores_dpr.items():
            print(key + ':\t' + str(np.mean(values)))

for key, values in tt_rel_scores_dpr.items():
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-tt_rel_scores_dpr-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(tt_rel_scores_dpr, f)


# 3.2 Recommendation score
with open('../recsum_/dump/nr-ft-1.2/args.pkl', 'rb') as f:
    args_rc = pickle.load(f)
    args_rc.recommender_ckpt_path = '/data/home/pengshancai/workspace/PLMNR_/dump/t-1.7.1/epoch-3-30000.pt'

recommender, tokenizer_rec = load_recommender(args_rc, args_rc.recommender_type)
recommender.eval()
recommender.to(device)

rcmd_scores = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    history = [title for title in rec['history'].split('; ')]
    for key in gen_results.keys():
        if key not in target_exp_names:
            continue
        headline = gen_results[key][idx]
        rcmd_score = get_rcmd_score(history, headline, recommender, tokenizer_rec, device)
        if math.isnan(rcmd_score):
            assert False
        rcmd_scores[key].append(rcmd_score)
    if idx % 1000 == 0 and idx > 0:
        for key, values in rcmd_scores.items():
            print(key + ':\t' + str(np.mean(values)))

for key, values in rcmd_scores.items():
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-rcmd-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(rcmd_scores, f)

exit()

with open('/data/home/pengshancai/workspace/recsum_/data/newsroom/train-bm25_cache.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# 3.3.1 User KP - title relevance (BM25)
ut_rel_scores_bm25 = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    kp_user = rec['kps_user']
    if type(kp_user) == list:
        kp_user = kp_user[0]
    for key in gen_results.keys():
        if key not in target_exp_names or key == 'src':
            continue
        headline = gen_results[key][idx]
        ut_rel_scores_bm25[key].append(get_bm25_scores_for_docs(bm25, kp_user, [headline]))
    # if idx % 1000 == 0 and idx > 0:
    #     for key, values in ut_rel_scores_bm25.items():
    #         print(key + ':\t' + str(np.mean(values)))

for key in target_exp_names:
    values = ut_rel_scores_bm25[key]
    print(key + ':\t' + str(round(np.mean(values), 3)))

with open('../recsum_/results/newsroom/kp_%s/scores-ut_rel_scores_bm25-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(ut_rel_scores_bm25, f)

# 3.3.2 title-text relevance score (BM25)
tt_rel_scores_bm25 = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    src = gen_results['srcs'][idx]
    for key in gen_results.keys():
        if key not in target_exp_names or key == 'src':
            continue
        headline = gen_results[key][idx]
        tt_rel_scores_bm25[key].append(get_bm25_scores_for_docs(bm25, headline, [src]))
    if idx % 1000 == 0 and idx > 0:
        for key, values in tt_rel_scores_bm25.items():
            print(key + ':\t' + str(np.mean(values)))

for key in target_exp_names:
    values = tt_rel_scores_bm25[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-tt_rel_scores_bm25-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(tt_rel_scores_bm25, f)


# 3.4 FactCC
checkpoint = '/data/home/pengshancai/workspace/recsum_/dump/factcc/factcc-checkpoint/'
factcc = BertForSequenceClassification.from_pretrained(checkpoint)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
factcc.to(device)
factcc.eval()

factcc_base_path = '../recsum_/results/newsroom/kp_%s/factcc/' % kp_version
if not os.path.exists(factcc_base_path):
    os.mkdir(factcc_base_path)

factcc_results = {}
for key in gen_results.keys():
    factcc_exp_path = factcc_base_path + key + '%s.json' % version
    srcs = gen_results['srcs']
    claims = gen_results[key]
    build_factcc_data(srcs, claims, factcc_exp_path)
    preds = evaluate(factcc, tokenizer, device, factcc_exp_path)
    factcc_results[key] = preds

for key in target_exp_names:
    values = factcc_results[key]
    print(key + ':\t' + str(100 * (sum(values)/len(values))))

with open('../recsum_/results/newsroom/kp_%s/scores-factcc-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(factcc_results, f)

exit()


# 3.5.1 User KP - title relevance (SBERT)
sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
ut_rel_scores_sbert = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    kp_user = rec['kps_user']
    if type(kp_user) == list:
        kp_user = kp_user[0]
    for key in gen_results.keys():
        if key not in target_exp_names or key == 'srcs':
            continue
        headline = gen_results[key][idx]
        ut_rel_scores_sbert[key].append(get_sbert_score(kp_user, headline, sbert))
    if idx % 1000 == 0 and idx > 0:
        for key, values in ut_rel_scores_sbert.items():
            print(key + ':\t' + str(np.mean(values)))

for key in target_exp_names:
    values = ut_rel_scores_sbert[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-ut_rel_scores_sbert-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(ut_rel_scores_bm25, f)


# 3.5.2 title-text relevance score (SBERT)
tt_rel_scores_sbert = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    src = gen_results['srcs'][idx]
    for key in gen_results.keys():
        if key not in target_exp_names or key == 'srcs':
            continue
        headline = gen_results[key][idx]
        tt_rel_scores_sbert[key].append(get_sbert_score(src, headline, sbert))
    if idx % 1000 == 0 and idx > 0:
        for key, values in tt_rel_scores_sbert.items():
            print(key + ':\t' + str(np.mean(values)))

for key in target_exp_names:
    values = tt_rel_scores_sbert[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-tt_rel_scores_sbert-%s.pkl' % (kp_version, version), 'wb') as f:
    pickle.dump(tt_rel_scores_sbert, f)

exit()






# # 4 CLM Perplexity
# ppl = defaultdict(list)
# progress = tqdm(range(len(dataset)))
# num_nan = 0
# for idx, rec in enumerate(dataset):
#     _ = progress.update(1)
#     ppl_instance = {}
#     use_instance = True
#     for key in gen_results.keys():
#         if key not in target_exp_names:
#             continue
#         headline = gen_results[key][idx]
#         ppl_instance[key] = get_clm_perplexity(headline, tokenizer_gpt2, gpt2, device)
#         if math.isnan(ppl_instance[key]):
#             use_instance = False
#             break
#     if use_instance:
#         for key, value in ppl_instance.items():
#             ppl[key].append(value)
#     if idx % 1000 == 0 and idx > 0:
#         for key, values in ppl.items():
#             print(key + ':\t' + str(np.mean(values)))
#         print('num_nan:\%s' % num_nan)
#
# for key, values in ppl.items():
#     print(key + ':\t' + str(np.mean(values)))
#
#
# with open('../recsum_/results/newsroom/ppl-%s.json' % version, 'wb') as f:
#     pickle.dump(ppl, f)


# MLM perplexity

mppl = defaultdict(list)
progress = tqdm(range(len(dataset)))
num_nan = 0
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    mppl_instance = {}
    for key in gen_results.keys():
        if key not in target_exp_names:
            continue
        headline = gen_results[key][idx]
        mppl[key].append(get_mlm_perplexity(headline, tokenizer_bert, bert, device))
    if idx % 1000 == 0 and idx > 0:
        for key, values in mppl.items():
            print(key + ':\t' + str(np.mean(values)))

for key, values in mppl.items():
    print(key + ':\t' + str(np.mean(values)))

with open('../recsum_/results/newsroom/kp_%s/scores-mppl-%s.json' % (kp_version, version), 'wb') as f:
    pickle.dump(mppl, f)


# 3.0.1 Length
lens = defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    for i, key in enumerate(gen_results.keys()):
        if key not in target_exp_names:
            continue
        lens[key].append(len(gen_results[key][idx].split(' ')))

for key in target_exp_names:
    values =lens[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))



# 3.0.2 Coverage / Density / Compression
coverage_scores, density_scores, compression_scores = defaultdict(list), defaultdict(list), defaultdict(list)
progress = tqdm(range(len(dataset)))
for idx, rec in enumerate(dataset):
    _ = progress.update(1)
    src = gen_results['srcs'][idx]
    for key in gen_results.keys():
        if key not in target_exp_names:
            continue
        headline = gen_results[key][idx]
        frag = Fragments(headline, src)
        coverage_scores[key].append(frag.coverage())
        density_scores[key].append(frag.density())
        compression_scores[key].append(frag.compression())
    if idx % 500 == 0 and idx > 0:
        print('----------\nCoverage:')
        for key, values in coverage_scores.items():
            print("%s:\t%s" % (key, np.mean(values)))
        print('----------\nDensity:')
        for key, values in density_scores.items():
            print("%s:\t%s" % (key, np.mean(values)))
        print('----------\nCompression:')
        for key, values in compression_scores.items():
            print("%s:\t%s" % (key, np.mean(values)))

# for key, values in coverage_scores.items():
for key in target_exp_names:
    values = coverage_scores[key]
    print("%s:\t%s" % (key, round(np.mean(values), 2)))

# for key, values in density_scores.items():
for key in target_exp_names:
    values = density_scores[key]
    print("%s:\t%s" % (key, round(np.mean(values), 2)))

# for key, values in compression_scores.items():
for key in target_exp_names:
    values = compression_scores[key]
    print("%s:\t%s" % (key, round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-abstractiveness-%s.pkl' % (kp_version, version), 'wb') as f:
    info = {
        'coverage_scores': coverage_scores,
        'density_scores': density_scores,
        'compression_scores': compression_scores,
    }
    pickle.dump(info, f)




# FactCC
# for key in gen_results.keys():
#

# 3.1.3 User KP - title relevance (ColBERT)
# ut_rel_scores_dpr = defaultdict(list)
# progress = tqdm(range(len(dataset)))
# for idx, rec in enumerate(dataset):
#     _ = progress.update(1)
#     kp_user = rec['kps_user'][0]
#     src, gold, pred_base = preds_base[idx]
#     pred_prompt_1 = preds_prompt_1[idx][2]
#     pred_naive_1 = preds_naive_1[idx][2]
#     pred_random_1 = preds_random_1[idx][2]
#     pred_prompt_2 = preds_prompt_2[idx][2]
#     pred_naive_2 = preds_naive_2[idx][2]
#     pred_random_2 = preds_random_2[idx][2]
#     pred_prompt_3 = preds_prompt_3[idx][2]
#     pred_naive_3 = preds_naive_3[idx][2]
#     pred_random_3 = preds_random_3[idx][2]
#     pred_prompt_4 = preds_prompt_4[idx][2]
#     pred_naive_4 = preds_naive_4[idx][2]
#     pred_random_4 = preds_random_4[idx][2]
#     pred_gold = preds_gold[idx][2]
#     # ut_rel_scores_dpr['score_original'].append(get_ut_rel_score_dpr(kp_user, gold))
#     # ut_rel_scores_dpr['score_pred_base'].append(get_ut_rel_score_dpr(kp_user, pred_base))
#     ut_rel_scores_dpr['score_pred_prompt'].append(get_ut_rel_score_dpr(kp_user, pred_prompt))
#     ut_rel_scores_dpr['score_pred_naive'].append(get_ut_rel_score_dpr(kp_user, pred_naive))
#     ut_rel_scores_dpr['score_pred_random'].append(get_ut_rel_score_dpr(kp_user, pred_random))
#     if idx % 1000 == 0 and idx > 0:
#         for key, values in ut_rel_scores_dpr.items():
#             print(key + ':\t' + str(np.mean(values)))
#
# for key, values in ut_rel_scores_dpr.items():
#     print(key + ':\t' + str(np.mean(values)))
#






