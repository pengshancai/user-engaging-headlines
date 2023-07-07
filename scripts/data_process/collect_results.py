import json
import jsonlines
import os

# Generated headlines
corpus = "newsroom-nr"     # "gigaword-gw"
path_in = '../recsum_/results/%s-pt-3.3/' % corpus
path_out = '../recsum_/results/_kaiqiang/%s/' % corpus.split('-')[0]
fnames = [fname for fname in os.listdir(path_in) if fname.endswith('.json')]

for i, fname in enumerate(fnames):
    with open(path_in + fname) as f:
        con = json.load(f)
    headlines = [rec[2] for rec in con]
    with open(path_out + 'list_A_%s' % fname, 'w') as f:
        json.dump(headlines, f)

with open(path_in + fnames[0]) as f:
    con = json.load(f)
    headlines = [rec[1] for rec in con]
    with open(path_out + 'list_A_original.json', 'w') as f:
        json.dump(headlines, f)

# Corpus headlines
corpus = 'newsroom'
path_in = '../recsum_/data/%s/' % corpus
path_out = '../recsum_/results/_kaiqiang/%s/' % corpus
fnames = [fname for fname in os.listdir(path_in) if fname.endswith('jsonl')]
for fname in fnames:
    headlines = []
    with jsonlines.open(path_in + fname) as f:
        for i, line in enumerate(f):
            headlines.append(line['title'])
    with open(path_out + 'list_B_%s' % fname, 'w') as f:
        json.dump(headlines, f)

corpus = 'gigaword'
path_in = '../recsum_/data/%s/' % corpus
path_out = '../recsum_/results/_kaiqiang/%s/' % corpus
fnames = [fname for fname in os.listdir(path_in) if fname.endswith('jsonl')]
for fname in fnames:
    headlines = []
    with jsonlines.open(path_in + fname) as f:
        for i, line in enumerate(f):
            headlines.append(line['headline'])
    with open(path_out + 'list_B_%s' % fname, 'w') as f:
        json.dump(headlines, f)


