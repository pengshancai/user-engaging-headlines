from collections import defaultdict
import re
import numpy as np
import matplotlib.pyplot as plt

exp_id = 'nr-ft-1.1'
path = '../recsum_/dump/%s/log.txt' % exp_id
with open(path) as f:
    lines = f.readlines()

pat_scores = re.compile('\tSCORE REC:\t([0-9\.]+)\tSCORE REL:\t([0-9\.]+)')

idx_ckpt = 0
scores_rec, scores_rel = defaultdict(list), defaultdict(list)
for line in lines:
    if line.startswith('*** New checkpoint reached ***'):
        idx_ckpt += 1
    se = pat_scores.search(line)
    if se:
        score_rec, score_rel = se.groups()
        score_rec, score_rel = float(score_rec), float(score_rel)
        scores_rec[idx_ckpt].append(score_rec)
        scores_rel[idx_ckpt].append(score_rel)

scores_rec_mean, scores_rel_mean = [], []
for idx_ckpt in scores_rec.keys():
    scores_rec_ckpt = scores_rec[idx_ckpt]
    scores_rel_ckpt = scores_rel[idx_ckpt]
    if not len(scores_rec_ckpt) < 100:
        print('Score Rec: %s:\t%s' % (idx_ckpt, np.mean(scores_rec_ckpt)))
        scores_rec_mean.append(np.mean(scores_rec_ckpt))
    if not len(scores_rel_ckpt) < 100:
        print('Score Rel: %s:\t%s' % (idx_ckpt, np.mean(scores_rel_ckpt)))
        scores_rel_mean.append(np.mean(scores_rel_ckpt))

ids_ckpt = list(range(len(scores_rec_mean)))
plt.plot(ids_ckpt, scores_rec_mean, color='r', label='REC Score')
plt.plot(ids_ckpt, scores_rel_mean, color='b', label='REL Score')
plt.xlabel('Checkpoints')
plt.ylabel('Score')
plt.legend(loc='best')
plt.title(exp_id)
plt.savefig('../recsum_/za/imgs/%s.png' % exp_id)
plt.show()

# for i, line in enumerate(f):
#     if i == 510:
#         break
#
#
# pass


