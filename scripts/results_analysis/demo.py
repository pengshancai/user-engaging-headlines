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


with open('../recsum_/dump/nr-pt-3.0/args.pkl', 'rb') as f:
    args_base = pickle.load(f)

summarizer, tokenizer = load_summarizer_naive(args_base, added_tokens=['<u>'])
model_base = SummarizerPreTrainNaive(args_base, summarizer, tokenizer)
ckpt_base_path = '../recsum_/dump/nr-pt-3.0/lightning_logs/version_0/'
ckpt_base = load_ckpt(ckpt_base_path)
model_base.load_state_dict(ckpt_base['state_dict'])
summarizer_base = model_base.summarizer
summarizer_base.to(device)

with open('../recsum_/dump/nr-pt-3.1-large/args.pkl', 'rb') as f:
    args_prompt = pickle.load(f)

summarizer, tokenizer_summ = load_summarizer_naive(args_prompt)
model_prompt = SummarizerPreTrainNaive(args_prompt, summarizer, tokenizer_summ)
ckpt_prompt_path = '../recsum_/dump/nr-pt-3.1-large/lightning_logs/version_0/'
ckpt_prompt = load_ckpt(ckpt_prompt_path)
model_prompt.load_state_dict(ckpt_prompt['state_dict'])
summarizer_prompt = model_prompt.summarizer
summarizer_prompt.to(device)
summarizer_prompt.eval()

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


idx = 100

rec = dataset[idx]
src = rec['src']
tgt = rec['tgt']
kps = rec['kps_doc']

user_titles = rec['history'].split('; ')
kps_focus = "; ".join(select_kps_early(kps, user_titles, encoder_kp, encoder_title, tokenizer_emb, 'max', 3))
inputs_prompt = tokenizer_summ(kps_focus + '</s> ' + src, return_tensors='pt', max_length=512).to(device)
outputs_prompt = summarizer_prompt.generate(**inputs_prompt).cpu()
pred_prompt = tokenizer_summ.decode(outputs_prompt.numpy()[0], skip_special_tokens=True)

inputs_base = tokenizer(src, return_tensors='pt', max_length=512).to(device)
outputs_base = summarizer.generate(**inputs_base).cpu()
pred_base = tokenizer_summ.decode(outputs_base.numpy()[0], skip_special_tokens=True)


print(kps_focus)
print('KP_User:\t%s' % rec['kps_user'])
print('Pred:\t\t%s' % pred_prompt)
print('Base:\t\t%s' % pred_base)
print('Original:\t%s' % rec['tgt'])

idx += 1


