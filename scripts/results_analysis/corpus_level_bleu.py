from datasets import load_from_disk
import os
import numpy as np

# Small model results
path = "../recsum_/results/newsroom/scores/newsroom_corpus_bleu/"
fnames = [fname for fname in os.listdir(path) if fname.startswith('newsroom')]
for fname in fnames:
    ds = load_from_disk(path + fname)
    bleu_max = np.mean([ds[i]['bleu_max'] for i in range(len(ds))])
    bleu_avg = np.mean([ds[i]['bleu_avg'] for i in range(len(ds))])
    print('%s\tbleu max: %s\tbleu avg: %s' % (fname, round(bleu_max * 100, 3), round(bleu_avg * 100, 3)))

# Big model
path = "../recsum_/results/newsroom/scores/newsroom_prompt_bleu/"
fnames = [fname for fname in os.listdir(path) if fname.startswith('newsroom')]
for fname in fnames:
    ds = load_from_disk(path + fname)
    bleu_max = np.mean([ds[i]['bleu_max'] for i in range(len(ds))])
    bleu_avg = np.mean([ds[i]['bleu_avg'] for i in range(len(ds))])
    print('%s\tbleu max: %s\tbleu avg: %s' % (fname, round(bleu_max * 100, 3), round(bleu_avg * 100, 3)))

fname = 'newsroom_kp-early-ft_bleu'
path_ = "../recsum_/results/newsroom/scores/newsroom_corpus_bleu/"
ds_ = load_from_disk(path_ + fname)
ds_top = [ds_[i] for i in range(100)]
bleu_max = np.mean([ds_top[i]['bleu_max'] for i in range(len(ds_top))])
bleu_avg = np.mean([ds_top[i]['bleu_avg'] for i in range(len(ds_top))])
print('%s\tbleu max: %s\tbleu avg: %s' % (fname, round(bleu_max * 100, 3), round(bleu_avg * 100, 3)))
