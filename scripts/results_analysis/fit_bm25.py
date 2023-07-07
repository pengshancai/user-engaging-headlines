from nltk import word_tokenize
import json
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm


path_nw = '/data/home/pengshancai/workspace/recsum_/data/newsroom/train.jsonl'
path_nr_cache = '/data/home/pengshancai/workspace/recsum_/data/newsroom/train-tokenized_cache.pkl'


def get_corpus(data_path):
    with open(data_path) as f:
        lines = f.readlines()
    corpus = []
    progress = tqdm(range(len(lines)))
    for line in lines:
        _ = progress.update(1)
        rec = json.loads(line)
        corpus.append(word_tokenize(rec['text'].lower()))
    return corpus


corpus = get_corpus(path_nw)
# with open(path_nr_cache, 'wb') as f:
#     pickle.dump(corpus, f)


with open(path_nr_cache, 'rb') as f:
    corpus = pickle.load(f)

bm25 = BM25Okapi(corpus)
bm25_path = '/data/home/pengshancai/workspace/recsum_/data/newsroom/train-bm25_cache.pkl'
with open(bm25_path, 'wb') as f:
    pickle.dump(bm25, f)
