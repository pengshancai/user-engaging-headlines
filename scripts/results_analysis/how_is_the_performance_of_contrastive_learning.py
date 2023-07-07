import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import pickle
from tqdm import tqdm
from models.selector import load_selector, Selector

"""
Preparation
"""
with open('../recsum_/dump/nr-sl-2.0/args.pkl', 'rb') as f:
    args = pickle.load(f)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
Load Model
"""


def load_ckpt(ckpt_path, partition_key='module.'):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


encoder_user, encoder_kp, tokenizer = load_selector(args)
model = Selector(args, encoder_user, encoder_kp, tokenizer)
ckpt_base_path = '/data/home/pengshancai/workspace/recsum_/dump/nr-sl-2.0/lightning_logs/version_2/checkpoints/'
ckpt_name = 'last.ckpt'
ckpt_path = ckpt_base_path + ckpt_name + '/checkpoint/mp_rank_00_model_states.pt'
ckpt = load_ckpt(ckpt_path)
model.load_state_dict(ckpt['state_dict'])
model.to(device)
encoder_user = model.encoder_user
encoder_kp = model.encoder_kp
encoder_user.eval()
encoder_kp.eval()


"""
Load Data
"""
split = 'dev'
with open('../recsum_/data/newsroom/kp_1.0/%s-kp-history_1.3.1.json' % split, 'r') as f:
    users = json.load(f)['data']

# with open('../recsum_/data/newsroom/%s-kp-history_1.2.json' % split, 'r') as f:
#     users = json.load(f)['data']


def get_scores(kp, histories_batch):
    inputs_histories = tokenizer(histories_batch, return_tensors='pt', padding=True, truncation=True).to(device)
    inputs_kp = tokenizer([kp], return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        user_vectors = encoder_user(**inputs_histories).pooler_output
        kp_vectors = encoder_kp(**inputs_kp).pooler_output
    logits = torch.matmul(kp_vectors, user_vectors.T)
    scores = F.softmax(logits, dim=1).cpu().numpy()
    return scores.reshape(-1)


def get_pos_rank(scores):
    scores = list(scores)
    ranks = list(np.argsort(scores))
    ranks.reverse()
    pos_rank = ranks.index(0)
    return pos_rank


NUM_TEST = 3000
NUM_NEG = 63
selected_users = random.sample(users, NUM_TEST)
pos_ranks = []
progress = tqdm(selected_users)
for user in selected_users:
    kps_user = user['kps']
    kp = random.sample(kps_user, 1)[0]
    users_negative_ = random.sample(users, NUM_NEG * 3)
    histories_negative = []
    for user_ in users_negative_:
        if kp not in user_['kps']:
            histories_negative.append(user_['history'])
        if len(histories_negative) >= NUM_NEG:
            break
    histories_batch = [user['history']] + histories_negative
    scores = get_scores(kp, histories_batch)
    pos_rank = get_pos_rank(scores)
    pos_ranks.append(pos_rank)
    _ = progress.update(1)

mean_rank = np.mean(pos_ranks)
# 13.87 (old-1.1 - Mixed user)
# 2.55 (old-1.2 - Pure user)
# 13.12 (new-1.1 - Mixed user)
# 2.53 (new-1.2 - Pure user)
top_5_hit = np.sum([1 for pos_rank in pos_ranks if pos_rank < 5]) / len(pos_ranks)
# 44.03 (old-1.1 - Mixed titles)
# 84.73 (old-1.2 - Pure user)
# 46.10 (new-1.1 - Mixed titles)
# 85.33 (new-1.2 - Pure user)
top_10_hit = np.sum([1 for pos_rank in pos_ranks if pos_rank < 10]) / len(pos_ranks)
# 56.56 (old-1.1 - Mixed titles)
# 92.83 (old-1.2 - Pure user)
# 58.33 (new-1.1 - Mixed titles)
# 92.90 (new-1.2 - Pure user)

print(mean_rank)
print(top_5_hit)
print(top_10_hit)
