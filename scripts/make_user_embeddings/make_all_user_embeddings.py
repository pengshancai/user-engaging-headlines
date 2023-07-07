import os
import pickle
import torch
from tqdm import tqdm
from utils.data_utils_kaiqiang import DataSetNRTitle
from models.recommender import load_recommender

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
processed_data_path = os.path.join("../recsum_/data/processed", 'processed-%s.pkl' % 'facebook-bart-base')
with open(processed_data_path, 'rb') as f:
    processed = pickle.load(f)
    news_index = processed['news_index']
    titles = processed['titles']
    logs = processed['logs']

dataset_train = DataSetNRTitle(news_index, titles, logs, args.npratio, args.user_log_length, device)
all_user_embs = []
NUM_USERS = 500000
progress = tqdm(range(NUM_USERS))
for i, idx in enumerate(range(NUM_USERS)):
    log_ids, log_mask = dataset_train.get_user_logs(idx)
    with torch.no_grad():
        user_emb = recommender.get_user_emb(log_ids, log_mask)
    all_user_embs.append(user_emb.cpu().numpy())
    _ = progress.update(1)

torch.save(all_user_embs, '../recsum_/data/processed/all_user_embs-%s.pt' % NUM_USERS)





