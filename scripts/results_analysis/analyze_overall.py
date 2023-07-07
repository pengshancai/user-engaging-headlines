import pickle
import os
import numpy as np

SCORES_PATH = '../recsum_/results/newsroom-nr-pt-3.3/scores/'
metrics = ['user-title relevance dpr', 'user-title relevance sbert', 'recommendation scores',
           'title-text relevance_dpr', 'title-text relevance sbert', 'title-text consistency factcc',
           'coverage_scores', 'density_scores', 'compression_scores', 'length', 'rouge1', 'rouge2', 'rougeL']


def round_results(scores):
    score = np.mean(scores)
    if score > 5:
        return round(np.mean(scores), 2)
    else:
        return round(np.mean(scores), 4)


"""
1. Comparing headlines obtained by different approaches
"""
fnames = [fname for fname in os.listdir(SCORES_PATH) if fname.endswith('pkl')]
for fname in fnames:
    with open(SCORES_PATH + fname, 'rb') as f:
        all_scores = pickle.load(f)
    exp_name = fname.split('.')[0]
    print(exp_name)
    for metric in metrics:
        scores = all_scores[metric]
        print('%s:\t%s' % (metric, round_results(scores)))
    print("-"*60)

"""
1.1 Comparing headlines obtained by previous approaches
"""
for fname in ['kp-early-ft.pkl']:
    with open(SCORES_PATH + fname, 'rb') as f:
        all_scores = pickle.load(f)
    exp_name = fname.split('.')[0]
    print(exp_name)
    for metric in metrics:
        scores = all_scores[metric]
        print('%s:\t%s' % (metric, round_results(scores)))
    print("-"*60)

"""
2. Comparing headlines generated for users of different number of core interests
"""
fname = 'kp-early-ft-1_additional_2.pkl'
with open(SCORES_PATH + fname, 'rb') as f:
    all_scores = pickle.load(f)

for metric in metrics:
    scores = all_scores[metric]
    scores_sections = [scores[i*2000: (i+1)*2000] for i in range(3)]
    for i, scores_section in enumerate(scores_sections):
        print('Users of %s core KPs - %s:\t%s' % ((i+1)*10, metric, round_results(scores_section)))





