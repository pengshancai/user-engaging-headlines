import json
import pickle
import numpy as np
import torch
from models.recommender import load_recommender
from utils.reward_utils import RecommenderScorer

split = 'dev'
MAX_LEN_RECOMMENDER = 24
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('../recsum_/data/newsroom/%s-synthesized-history.json' % split, 'r') as f:
    histories_all = json.load(f)

with open('../recsum_/data/newsroom/dev-titles.json', 'r') as f:
    titles_dev = json.load(f)

with open('../recsum_/data/newsroom/train-titles.json', 'r') as f:
    titles_train = json.load(f)

with open('../recsum_/dump/nr-ft-1.2/args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.recommender_ckpt_path = '/data/home/pengshancai/workspace/PLMNR_/dump/t-1.7.1/epoch-3-30000.pt'

recommender, tokenizer_rec = load_recommender(args, args.recommender_type)
recommender.eval()
recommender.to(device)


def get_user_emb(user_histories):
    histories_encoded = tokenizer_rec(user_histories, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_RECOMMENDER)
    histories_ids, histories_mask = histories_encoded['input_ids'].to(device), histories_encoded['attention_mask'].to(device)
    with torch.no_grad():
        input_ids = torch.cat((histories_ids, histories_mask), dim=1)  # Required format of the recommendation model
        log_mask = torch.ones(input_ids.shape[0], dtype=int)
        user_features = torch.unsqueeze(input_ids.to(device), 0)
        log_mask = torch.unsqueeze(log_mask.to(device), 0)
        user_emb = recommender.get_user_emb(user_features, log_mask)
    return user_emb


def get_title_emb(title):
    titles_encoded = tokenizer_rec([title], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_RECOMMENDER)
    titles_ids, titles_mask = titles_encoded['input_ids'].to(device), titles_encoded['attention_mask'].to(device)
    with torch.no_grad():
        input_ids = torch.cat((titles_ids, titles_mask), dim=1)  # Required format of the recommendation model
        titles_embs = recommender.get_news_emb(torch.unsqueeze(input_ids, dim=0)).squeeze(0)
    return titles_embs


idx_title = 5
title = titles_dev[idx_title]
users_histories = histories_all[str(idx_title)]
# kws = ['Pharmaceuticals', 'Women and Girls', 'Medicine and Health', 'Vaccines Immunization']
kw1 = 'Women and Girls'
title1 = 'Medical breakthrough will save millions of women from breast cancer'

kw2 = 'Vaccines Immunization'
title2 = 'Study shows the new breast cancer vaccine looks safe'  # original title
user_histories_1 = [titles_train[idx] for idx in users_histories[kw1]]
user_histories_2 = [titles_train[idx] for idx in users_histories[kw2]]

title_emb1 = get_title_emb(title1).cpu().numpy()[0]  # "Women and Girls" title
user_emb1 = get_user_emb(user_histories_1).cpu().numpy()[0]  # "Women and Girls" user

title_emb2 = get_title_emb(title2).cpu().numpy()[0]  # "Vaccines Immunization" title
user_emb2 = get_user_emb(user_histories_2).cpu().numpy()[0]  # "Vaccines Immunization" user

score_1_1 = np.dot(user_emb1, title_emb1)  # 1.5421964
score_1_2 = np.dot(user_emb1, title_emb2)  # 1.414763
print(score_1_1)
print(score_1_2)
score_2_1 = np.dot(user_emb2, title_emb1)  # 3.2450318
score_2_2 = np.dot(user_emb2, title_emb2)  # 3.978878
print(score_2_1)
print(score_2_2)


idx_title = 8
title = titles_dev[idx_title]
users_histories = histories_all[str(idx_title)]
# kws = ['Victoria and Albert Museum', 'Tate Modern', 'London', 'Photography']
kw1 = 'London'
title1 = 'What to do in London this weekend'
user_histories_1 = [titles_train[idx] for idx in users_histories[kw1]]
kw2 = 'Photography'
title2 = 'Political, provocative, personal: photography to look forward to'
user_histories_2 = [titles_train[idx] for idx in users_histories[kw2]]

title_emb1 = get_title_emb(title1).cpu().numpy()[0]  # "Women and Girls" title
user_emb1 = get_user_emb(user_histories_1).cpu().numpy()[0]  # "Women and Girls" user

title_emb2 = get_title_emb(title2).cpu().numpy()[0]  # "Vaccines Immunization" title
user_emb2 = get_user_emb(user_histories_2).cpu().numpy()[0]  # "Vaccines Immunization" user

score_1_1 = np.dot(user_emb1, title_emb1)  # 1.552331
score_1_2 = np.dot(user_emb1, title_emb2)  # 2.041475
print(score_1_1)
print(score_1_2)
score_2_1 = np.dot(user_emb2, title_emb1)  # 1.0913637
score_2_2 = np.dot(user_emb2, title_emb2)  # 1.9895437
print(score_2_1)
print(score_2_2)



idx_title = 10
title = titles_dev[idx_title]
users_histories = histories_all[str(idx_title)]
print(users_histories.keys())
# kws = ['Music', 'Boston (Mass)']
kw1 = 'Music'
title1 = 'Music review: Jake Bugg at the House of Blues'
user_histories_1 = [titles_train[idx] for idx in users_histories[kw1]]
kw2 = 'Boston (Mass)'
title2 = 'Jake Bugg will set foot on stage in Boston'
user_histories_2 = [titles_train[idx] for idx in users_histories[kw2]]

title_emb1 = get_title_emb(title1).cpu().numpy()[0]  # "Women and Girls" title
user_emb1 = get_user_emb(user_histories_1).cpu().numpy()[0]  # "Women and Girls" user

title_emb2 = get_title_emb(title2).cpu().numpy()[0]  # "Vaccines Immunization" title
user_emb2 = get_user_emb(user_histories_2).cpu().numpy()[0]  # "Vaccines Immunization" user

score_1_1 = np.dot(user_emb1, title_emb1)  # 4.1037664
score_1_2 = np.dot(user_emb1, title_emb2)  # 3.11593
print(score_1_1)
print(score_1_2)
score_2_1 = np.dot(user_emb2, title_emb1)  # 0.8166901
score_2_2 = np.dot(user_emb2, title_emb2)  # 1.0471029
print(score_2_1)
print(score_2_2)


idx_title = 27
title = titles_dev[idx_title]
users_histories = histories_all[str(idx_title)]
print(users_histories.keys())
# kws = ['Retail Stores and Trade', 'Wal-Mart Stores Inc', 'Whole Foods', 'Company Reports']
kw1 = 'Whole Foods'
title1 = 'Whole Foods takes aim at lower-end with new chain of stores'
user_histories_1 = [titles_train[idx] for idx in users_histories[kw1]]
kw2 = 'Wal-Mart Stores Inc'
title2 = 'Wal-Mart face challenges in retailing business'
user_histories_2 = [titles_train[idx] for idx in users_histories[kw2]]

title_emb1 = get_title_emb(title1).cpu().numpy()[0]  # "Women and Girls" title
user_emb1 = get_user_emb(user_histories_1).cpu().numpy()[0]  # "Women and Girls" user

title_emb2 = get_title_emb(title2).cpu().numpy()[0]  # "Vaccines Immunization" title
user_emb2 = get_user_emb(user_histories_2).cpu().numpy()[0]  # "Vaccines Immunization" user

score_1_1 = np.dot(user_emb1, title_emb1)  # 5.0631685
score_1_2 = np.dot(user_emb1, title_emb2)  # 4.365861
print(score_1_1)
print(score_1_2)
score_2_1 = np.dot(user_emb2, title_emb1)  # 6.2114425
score_2_2 = np.dot(user_emb2, title_emb2)  # 5.924905
print(score_2_1)
print(score_2_2)



idx_title = 90
title = titles_dev[idx_title]
users_histories = histories_all[str(idx_title)]
print(users_histories.keys())
# kws = ['Star Wars', 'Movies']
kw1 = 'Star Wars'
title1 = 'Star War comes back to cinema this weekend'
user_histories_1 = [titles_train[idx] for idx in users_histories[kw1]]
kw2 = 'Movies'
title2 = 'New movies worth your anticipation this weenkend'
user_histories_2 = [titles_train[idx] for idx in users_histories[kw2]]

title_emb1 = get_title_emb(title1).cpu().numpy()[0]
user_emb1 = get_user_emb(user_histories_1).cpu().numpy()[0]

title_emb2 = get_title_emb(title2).cpu().numpy()[0]
user_emb2 = get_user_emb(user_histories_2).cpu().numpy()[0]

score_1_1 = np.dot(user_emb1, title_emb1)  # 4.6427093
score_1_2 = np.dot(user_emb1, title_emb2)  # 4.201249
print(score_1_1)
print(score_1_2)
score_2_1 = np.dot(user_emb2, title_emb1)  # 4.4715233
score_2_2 = np.dot(user_emb2, title_emb2)  # 4.439481
print(score_2_1)
print(score_2_2)



idx_title = 180
title = titles_dev[idx_title]
users_histories = histories_all[str(idx_title)]
print(users_histories.keys())
# kws = ['Tesla', 'Fires', 'Automobile safety']
kw1 = 'Tesla'
title1 = 'Tesla Model S will be coming to the market after fire tests'
user_histories_1 = [titles_train[idx] for idx in users_histories[kw1]]
kw2 = 'Automobile safety'
title2 = 'Automobile safety is an important factor to consider when buying electric cars'
user_histories_2 = [titles_train[idx] for idx in users_histories[kw2]]

title_emb1 = get_title_emb(title1).cpu().numpy()[0]
user_emb1 = get_user_emb(user_histories_1).cpu().numpy()[0]

title_emb2 = get_title_emb(title2).cpu().numpy()[0]
user_emb2 = get_user_emb(user_histories_2).cpu().numpy()[0]

score_1_1 = np.dot(user_emb1, title_emb1)  # 5.780719
score_1_2 = np.dot(user_emb1, title_emb2)  # 3.6321836
print(score_1_1)
print(score_1_2)
score_2_1 = np.dot(user_emb2, title_emb1)  # 5.1947446
score_2_2 = np.dot(user_emb2, title_emb2)  # 4.946975
print(score_2_1)
print(score_2_2)








