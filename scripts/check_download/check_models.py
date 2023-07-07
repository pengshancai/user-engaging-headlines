import pickle
import torch
from models.summarizer import load_summarizer_naive
from models.general import SummarizerPreTrainNaive
from models.selector import load_selector, Selector
from models.recommender import load_recommender, NRModel, BartConfig, BartModel, AutoTokenizer

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
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


# ----------------------------------------------------------------------------------------------------------------------
"""
Summarizer Base
"""
with open('../recsum_/dump/nr-pt-3.0/args.pkl', 'rb') as f:
    args_base = pickle.load(f)
    args_base.summarizer_model_path = 'facebook/bart-large'

summarizer, tokenizer = load_summarizer_naive(args_base, added_tokens=['<u>'])
model_base = SummarizerPreTrainNaive(args_base, summarizer, tokenizer)
ckpt_base_path = '../recsum_/dump/nr-pt-3.0/lightning_logs/version_0/'
ckpt_base = load_ckpt(ckpt_base_path)
model_base.load_state_dict(ckpt_base['state_dict'])


# ----------------------------------------------------------------------------------------------------------------------
"""
Summarizer Prompt
"""
with open('../recsum_/dump/nr-pt-3.1-large/args.pkl', 'rb') as f:
    args_prompt = pickle.load(f)
    args_prompt.summarizer_model_path = 'facebook/bart-large'

summarizer, tokenizer_summ = load_summarizer_naive(args_prompt)
model_prompt = SummarizerPreTrainNaive(args_prompt, summarizer, tokenizer_summ)
ckpt_prompt_path = '../recsum_/dump/nr-pt-3.1-large/lightning_logs/version_0/'
ckpt_prompt = load_ckpt(ckpt_prompt_path)
model_prompt.load_state_dict(ckpt_prompt['state_dict'])

# ----------------------------------------------------------------------------------------------------------------------
"""
Selector 3.0
"""
with open('../recsum_/dump/nr-sl-3.0/args.pkl', 'rb') as f:
    args_sl = pickle.load(f)
    args_sl.dpr_question_encoder_path = 'facebook/dpr-question_encoder-single-nq-base'
    args_sl.dpr_ctx_encoder_path = 'facebook/dpr-ctx_encoder-single-nq-base'

encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_sl)
model_sl = Selector(args_sl, encoder_src, encoder_tgt, tokenizer_emb)
ckpt_sl_path = '../recsum_/dump/nr-sl-3.0/lightning_logs/version_2/'
ckpt_sl = load_ckpt(ckpt_sl_path)
model_sl.load_state_dict(ckpt_sl['state_dict'])


"""
Selector 2.0
"""
with open('../recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
    args_sl = pickle.load(f)
    args_sl.dpr_question_encoder_path = 'facebook/dpr-question_encoder-single-nq-base'
    args_sl.dpr_ctx_encoder_path = 'facebook/dpr-ctx_encoder-single-nq-base'

encoder_src, encoder_tgt, tokenizer_emb = load_selector(args_sl)
model_sl = Selector(args_sl, encoder_src, encoder_tgt, tokenizer_emb)
ckpt_sl_path = '../recsum_/dump/nr-sl-2.0/lightning_logs/version_3/'
ckpt_sl = load_ckpt(ckpt_sl_path)
model_sl.load_state_dict(ckpt_sl['state_dict'])


"""
Recommender
"""
with open('../recsum_/dump/nr-ft-1.2/args.pkl', 'rb') as f:
    args_rc = pickle.load(f)
    args_rc.recommender_model_path = 'facebook/bart-base'
    args_rc.recommender_ckpt_path = '../PLMNR_/dump/t-1.7.1/epoch-3-30000.pt'

config = BartConfig.from_pretrained(args_rc.recommender_model_path, output_hidden_states=True)
full_model = BartModel.from_pretrained(args_rc.recommender_model_path, config=config)
encoder_model = full_model.encoder
recommender = NRModel(args_rc, encoder_model)
tokenizer = AutoTokenizer.from_pretrained(args_rc.recommender_model_path, use_fast=not args_rc.use_slow_tokenizer)
model_state_dict = torch.load(args_rc.recommender_ckpt_path, map_location=torch.device('cpu'))
if type(model_state_dict) == dict:
    model_state_dict = model_state_dict['model_state_dict']
recommender.load_state_dict(model_state_dict)
print('Model state dict loaded from %s' % args_rc.recommender_ckpt_path)


