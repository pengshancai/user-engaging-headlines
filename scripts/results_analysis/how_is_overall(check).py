import pickle
import numpy as np
from collections import defaultdict

kp_version = '7.0'
data_version = '1.3.1'

# exp_cates = ['late', 'early-max', 'early-avg', 'naive', 'random']
exp_cates = ['late', 'early-max', 'naive', 'random']
exp_names = ['originals', 'none-kp', 'gold_kps'] + \
            [exp_cate + '-1' for exp_cate in exp_cates] + \
            [exp_cate + '-2' for exp_cate in exp_cates] + \
            [exp_cate + '-3' for exp_cate in exp_cates] + \
            [exp_cate + '-4' for exp_cate in exp_cates] + \
            [exp_cate + '-5' for exp_cate in exp_cates]


class MinMaxNormalizer:
    def __init__(self, scores):
        self.scores = scores
        self.min = 100
        self.max = 0
        self.get_min_max()
        self.get_mean()
    def get_min_max(self):
        for key, values in self.scores.items():
            for value in values:
                if value < self.min:
                    self.min = value
                if value > self.max:
                    self.max = value
    def get_mean(self):
        scores = []
        for key, values in self.scores.items():
            scores += values
        self.mean = np.mean(scores)
    def get_normed_score(self, score):
        return (score-self.mean) / (self.max-self.min)
    def get_normed_scores(self):
        self.scores_normed = defaultdict(list)
        for key, values in self.scores.items():
            for value in values:
                self.scores_normed[key].append(self.get_normed_score(value))
        return self.scores_normed


class Standardlizer:
    def __init__(self, scores):
        self.scores = scores
        scores_all = []
        for key, values in self.scores.items():
            scores_all += values
        self.miu = np.mean(scores_all)
        self.sigma = np.std(scores_all)
    def get_stadard_scores(self):
        self.scores_standard = defaultdict(list)
        for key, values in self.scores.items():
            for value in values:
                self.scores_standard[key].append((value-self.miu)/self.sigma)
        return self.scores_standard


with open('../recsum_/results/newsroom/kp_%s/scores-ut_rel_scores_dpr-%s.pkl' % (kp_version, data_version), 'rb') as f:
    ut_rel_scores_dpr = pickle.load(f)

for key in exp_names:
    values = ut_rel_scores_dpr[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))

with open('../recsum_/results/newsroom/kp_%s/scores-rcmd-%s.pkl' % (kp_version, data_version), 'rb') as f:
    rcmd_scores = pickle.load(f)

for key in exp_names:
    values = rcmd_scores[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))

# normalizer = MinMaxNormalizer(rcmd_scores)
# rcmd_scores_normed = normalizer.get_normed_scores()
#
# standardlizer = Standardlizer(rcmd_scores)
# scores_std = standardlizer.get_stadard_scores()

with open('../recsum_/results/newsroom/kp_%s/scores-tt_rel_scores_dpr-%s.pkl' % (kp_version, data_version), 'rb') as f:
    tt_rel_scores_dpr = pickle.load(f)

for key in exp_names:
    values = tt_rel_scores_dpr[key]
    print(key + ':\t' + str(round(np.mean(values), 2)))




