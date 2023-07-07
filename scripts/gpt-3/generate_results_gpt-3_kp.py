import os
import openai
import json
import jsonlines
import time
import pickle
from models.selector import load_selector, Selector
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

api_key_path = '../recsum_/za/api_key.txt'
# output_path = "../recsum_/results/newsroom-nr-sl-3.5-text-davinci-003/kp-early-ft.txt"
data_path = "../recsum_/data/newsroom/synthesized_user/test.json"
selector_dump_dir = "../recsum_/dump/nr-sl-3.5/"
model_name = "text-davinci-003"
prompt_a = "Generate a headline for the following article focusing on %s: %s"
prompt_b = "Within ten words, generate a headline for the following article focusing on %s: %s"
prompt_c = "Generate a compelling headline for the following news article that a reader who has already read a series of articles on the topic of %s would find interesting: %s"
prompt_d = "Within ten words, generate a compelling headline for the following news article that a reader who has already read a series of articles on the topic of %s would find interesting: %s"

prompt_bx = "%s. Generate a compelling headline within ten words for the above news article focusing on %s."
prompt_dx = "%s. Generate a compelling headline within ten words for the above news article that a reader who has already read a series of articles on the topic of %s would find interesting."


prompt_base = prompt_dx
output_path = "../recsum_/results/newsroom-nr-sl-3.5-text-davinci-003/kp-early-ft-prompt-dx.json"
test_size = 100

top_k = 1
interval = 1
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open(api_key_path) as f:
    openai.api_key = f.read().strip()

with jsonlines.open(data_path) as f:
    con = [line for line in f]


def load_ckpt(ckpt_dir, partition_key='module.'):
    if 'last.ckpt' not in ckpt_dir:
        ckpt_dir = ckpt_dir + 'lightning_logs/version_0/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_dir)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


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


with open(selector_dump_dir + 'args.pkl', 'rb') as f:
    args_selector = pickle.load(f)

encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_selector)
model_sl = Selector(args_selector, encoder_src, encoder_tgt, tokenizer_emb)
ckpt_sl = load_ckpt(selector_dump_dir)
model_sl.load_state_dict(ckpt_sl['state_dict'])
encoder_kp, encoder_user = model_sl.encoder_src.to(device), model_sl.encoder_tgt.to(device)
encoder_kp.eval()
encoder_user.eval()
tokenizer_sum = AutoTokenizer.from_pretrained('facebook/bart-large', use_fast=True)


def get_prompt(rec, prompt_base):
    user_titles = rec['history'].split('; ')
    kps_focus = "; ".join(
        select_kps_early(rec['kps_doc'], user_titles, encoder_kp, encoder_user, tokenizer_emb, 'max', top_k))
    article = rec['src']
    article = tokenizer_sum.decode(tokenizer_sum.encode(article, truncation=True, max_length=512), skip_special_tokens=True)
    prompt = prompt_base % (article, kps_focus)
    return prompt


def generate_headline_gpt3(rec, prompt):
    try:
        response = openai.Completion.create(model=model_name, prompt=prompt, temperature=0, max_tokens=32)
        return response["choices"][0]['text'].strip().replace('\n', ' ')
    except:
        print('Server error, restart in 10s')
        time.sleep(10)
        return generate_headline_gpt3(rec, prompt)


def judge_time(previous):
    now = time.time()
    while now - previous < interval:
        time.sleep(1)
        now = time.time()
    return now


# def update_result_file(output_path, idx, prompt, headline):
#     with open(output_path, 'a') as f:
#         f.write('%s\t%s\t%s\n' % (idx, prompt, headline))


def get_processed_ids(output_path):
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = f.readlines()
        ids_current = set([int(result.split('\t')[0]) for result in results])
    else:
        return set([])
    return ids_current


test_set = con[:test_size]
progress = tqdm(range(len(test_set)))
previous = time.time() - 3
processed_ids = get_processed_ids(output_path)
results = []
for idx, rec in enumerate(test_set):
    if idx in processed_ids:
        _ = progress.update(1)
        continue
    previous = judge_time(previous)
    prompt = get_prompt(rec, prompt_base)
    headline = generate_headline_gpt3(rec, prompt)
    results.append([idx, prompt, headline])
    _ = progress.update(1)

with open(output_path, 'w') as f:
    json.dump(results, f)







