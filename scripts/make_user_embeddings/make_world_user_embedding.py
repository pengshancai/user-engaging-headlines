import os
import pickle
import torch
import random
import numpy as np
from tqdm import tqdm
from utils.data_utils_kaiqiang import DataSetNRTitle
from recycle.recommender import load_recommender

with open('../PLMNR_/za/args_rcmd/args_t-1.7.1.pkl', 'rb') as f:
    args = pickle.load(f)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on CUDA')
else:
    device = torch.device('cpu')
    print('Running on CPU')

dump_path = '../PLMNR_/dump/t-1.7.1/epoch-3-30000.pt'
recommender = load_recommender(args, 'bart-base-encoder', dump_path)
recommender.to(device)
processed_data_path = os.path.join(args.root_data_dir, 'mind/MINDlarge_%s/processed-%s.pkl' % ('train', 'facebook-bart-base'))
with open(processed_data_path, 'rb') as f:
    processed = pickle.load(f)
    news_index = processed['news_index']
    titles = processed['titles']
    logs = processed['logs']

dataset_train = DataSetNRTitle(news_index, titles, logs, args.npratio, args.user_log_length, device)
NUM_INS = 100000
ids_selected = random.sample(range(len(dataset_train)), NUM_INS)

user_embs = []
progress = tqdm(range(len(ids_selected)))
for i, idx in enumerate(ids_selected):
    log_ids, log_mask = dataset_train.get_user_logs(idx)
    with torch.no_grad():
        user_emb = recommender.get_user_emb(log_ids, log_mask)
    user_embs.append(user_emb)
    _ = progress.update(1)

user_emb = torch.cat(user_embs, dim=0)
world_user_emb = torch.mean(user_emb, dim=0)
torch.save(world_user_emb, '../recsum_/data/processed/world_user_emb.pt')


# Some results_analysis

# 1. Standard deviation
stds = []
for user_emb in user_embs:
    stds.append(np.std(torch.squeeze(user_emb).cpu().numpy()))

print('Average STD of user_emb: %s' % np.mean(stds))
print('STD of world_user_emb: %s' % np.std(world_user_emb.cpu().numpy()))

# 2. ABS
absolutes = []
for user_emb in user_embs:
    absolutes.append(np.mean(abs(torch.squeeze(user_emb).cpu().numpy())))

print('Average mean ABS of user_emb: %s' % np.mean(absolutes))
print('Mean ABS of world_user_emb: %s' % np.mean(abs(world_user_emb.cpu().numpy())))

