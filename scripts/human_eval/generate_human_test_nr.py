import csv
import json
from collections import defaultdict
import random
import pickle
import torch
from models.summarizer import load_summarizer
from models.selector import load_selector, Selector
from models.general import SummarizerPreTrainNaive
import copy


NUM_NEWS = 6
top_k = 1

"""
Step 0: Preparation
"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_ckpt(ckpt_dir, partition_key='module.'):
    if 'last.ckpt' not in ckpt_dir:
        ckpt_dir = ckpt_dir + 'lightning_logs/version_0/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_dir)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


def load_selector_early(args):
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



with open('../recsum_/data/newsroom/human_eval/kp_headlines_pool_nr.csv') as f:
    csvreader = csv.reader(f)
    lines = [line for i, line in enumerate(csvreader) if i >= 1]
    kp2headlines = defaultdict(list)
    for _, kp, headline in lines:
        kp2headlines[kp].append(headline)

with open('../recsum_/data/newsroom/human_eval/kp_texts_pool_nr.json') as f:
    textkp2info = json.load(f)

with open('../recsum_/za/args/args_sel.pkl', 'rb') as f:
    args = pickle.load(f)
    args.dataset_file = "../recsum_/data/newsroom/synthesized_user/test.json"
    args.max_src_len = 512
    args.max_tgt_len = 32

# early_ft
args_early_ft = copy.deepcopy(args)
args_early_ft.generator_dump_dir = '../recsum_/dump/nr-pt-3.3/'
args_early_ft.selector_dump_dir = '../recsum_/dump/nr-sl-3.5/'
args_early_ft.kp_select_method = 'early-ft'


# early_naive
args_early_naive = copy.deepcopy(args)
args_early_naive.generator_dump_dir = '../recsum_/dump/nr-pt-3.3/'
args_early_naive.selector_dump_dir = '../recsum_/dump/nr-sl-3.5/'
args_early_naive.kp_select_method = 'early-naive'

# early_naive
args_random = copy.deepcopy(args)
args_random.kp_select_method = 'random'

# none-kp
args_naive = copy.deepcopy(args)
args_naive.generator_dump_dir = '../recsum_/dump/nr-pt-3.0/'
args_naive.selector_dump_dir = None
args_naive.kp_select_method = 'none-kp'

# Load selectors
encoder_kp_ft, encoder_user_ft, tokenizer_emb_ft = load_selector_early(args_early_ft)
encoder_kp_nv, encoder_user_nv, tokenizer_emb_nv = load_selector_naive(args_early_naive)
# Load summarizers
summarizer_prompt, tokenizer_sum_prompt = load_summarizer_prompt(args_early_ft)
summarizer_direct, tokenizer_sum_direct = load_summarizer_naive(args_naive)


"""
Step 1: Define neccessary functions
"""


# with open('../recsum_/data/newsroom/human_eval/kp_headlines_pool.csv') as f:
#     csvreader = csv.reader(f)
#     lines = [line for i, line in enumerate(csvreader) if i >= 1]
#     kp2headlines = defaultdict(list)
#     for _, kp, headline in lines:
#         kp2headlines[kp].append(headline)

def get_user_history(path):
    kps_selected, headlines_selected = [], []
    with open(path) as f:
        csvreader = csv.reader(f)
        lines = [line for i, line in enumerate(csvreader) if i >= 1]
        for selected, kp, headline in lines:
            if selected != '':
                kps_selected.append(kp)
                headlines_selected.append(headline)
    return set(list(kps_selected)), headlines_selected


def select_random_news(kps_selected):
    kp_news = random.sample(kps_selected, 1)[0]
    headline_news, text_news, kps_cand = random.sample(textkp2info[kp_news], 1)[0]
    if kps_cand.endswith(';'):
        kps_cand = kps_cand[:-1]
    return kp_news, headline_news, text_news, kps_cand


def select_news_by_kp(kp_news):
    headline_news, text_news, kps_cand = random.sample(textkp2info[kp_news], 1)[0]
    if kps_cand.endswith(';'):
        kps_cand = kps_cand[:-1]
    return kp_news, headline_news, text_news, kps_cand


def generate_headline_prompt(args_, text_news, kps_cand, history, encoder_kp, encoder_user, tokenizer_emb, summarizer, tokenizer_sum, top_k_in):
    kps_focus = "; ".join(
        select_kps_early(kps_cand, history, encoder_kp, encoder_user, tokenizer_emb, select_method='max',
                         top_k=top_k_in))
    src_ = kps_focus + '</s> ' + text_news
    inputs = tokenizer_sum(src_, return_tensors='pt', max_length=args_.max_src_len, truncation=True).to(device)
    inputs['max_length'] = args_early_ft.max_tgt_len
    outputs = summarizer.generate(**inputs).cpu()
    pred = tokenizer_sum.decode(outputs.numpy()[0], skip_special_tokens=True)
    return pred, kps_focus


def generate_headline_random(args_, text_news, kps_cand, summarizer, tokenizer_sum, top_k_in):
    kps_focus = "; ".join(random.choices(kps_cand.split(';'), k=top_k_in))
    src_ = kps_focus + '</s> ' + text_news
    inputs = tokenizer_sum(src_, return_tensors='pt', max_length=args_.max_src_len, truncation=True).to(device)
    inputs['max_length'] = args_early_ft.max_tgt_len
    outputs = summarizer.generate(**inputs).cpu()
    pred = tokenizer_sum.decode(outputs.numpy()[0], skip_special_tokens=True)
    return pred, kps_focus


def generate_headline_direct(args_, text_news, summarizer, tokenizer_sum):
    src_ = text_news
    inputs = tokenizer_sum(src_, return_tensors='pt', max_length=args_.max_src_len, truncation=True).to(device)
    inputs['max_length'] = args.max_tgt_len
    outputs = summarizer.generate(**inputs).cpu()
    pred = tokenizer_sum.decode(outputs.numpy()[0], skip_special_tokens=True)
    return pred


def get_text_news(text_news, tokenizer_sum, args_):
    input_ids = tokenizer_sum(text_news, max_length=args_.max_src_len, truncation=True)['input_ids']
    text_news_ = tokenizer_sum.decode(input_ids, skip_special_tokens=True)
    text_news_ = text_news_.replace('\n\n', '\n')
    return text_news_


"""
Step 2: Get news articles to summarize
"""


def show_case(path, kp_in=None):
    kps_selected, history = get_user_history(path)
    if kp_in:
        kp_core, headline_news, text_news, kps_cand = select_news_by_kp(kp_in)
    else:
        kp_core, headline_news, text_news, kps_cand = select_random_news(kps_selected)
    print("All User KPs:\t%s" % kps_selected)
    print("Core KP:\t%s" % kp_core)
    """
    text
    """
    print('News:\t\t%s' % get_text_news(text_news, tokenizer_sum_direct, args_naive))
    print('\n')
    """
    Original
    """
    print(headline_news)
    """
    early-ft-1
    """
    pred_early_ft_1, kps_focus = generate_headline_prompt(args_early_ft, text_news, kps_cand, history, encoder_kp_ft,
                                                        encoder_user_ft, tokenizer_emb_ft, summarizer_prompt,
                                                        tokenizer_sum_prompt, 1)
    print('%s\t(KPs Focus:\t%s)' % (pred_early_ft_1, kps_focus))
    """
    early-naive-1
    """
    pred_early_nv_1, kps_focus = generate_headline_prompt(args_early_naive, text_news, kps_cand, history, encoder_kp_nv,
                                                        encoder_user_nv, tokenizer_emb_nv, summarizer_prompt,
                                                        tokenizer_sum_prompt, 1)
    print('%s\t(KPs Focus:\t%s)' % (pred_early_nv_1, kps_focus))
    """
    direct
    """
    pred_naive = generate_headline_direct(args_naive, text_news, summarizer_direct, tokenizer_sum_direct)
    print('%s' % pred_naive)
    """
    random-1
    """
    pred_random_1, kp_focus = generate_headline_random(args_random, text_news, kps_cand, summarizer_prompt,
                                                        tokenizer_sum_prompt, 1)
    print('%s\t(KPs Focus:\t%s)' % (pred_random_1, kp_focus))


path = '../recsum_/data/human/user_history/kp_headlines_pool_nr_pengshan.csv'
show_case(path, 'Adidas')


