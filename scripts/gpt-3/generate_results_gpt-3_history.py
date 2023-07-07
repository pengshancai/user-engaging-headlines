import os
import openai
import jsonlines
import time
import torch
from tqdm import tqdm

api_key_path = '../recsum_/za/api_key.txt'
data_path = "../recsum_/data/newsroom/synthesized_user/test.json"
selector_dump_dir = "../recsum_/dump/nr-sl-3.5/"
model_name = "text-davinci-003"
prompt_e = "Assume a reader has already read a series of articles titled \"%s\", … Here’s an input news article: \"%s\". Generate a compelling headline for this news article that the reader would find interesting."
prompt_f = "Assume a reader has already read a series of articles titled \"%s\", … Here’s an input news article: \"%s\". Generate a compelling headline (within ten words) for this news article that the reader would find interesting."
prompt = prompt_e
output_path = "../recsum_/results/newsroom-nr-sl-3.5-text-davinci-003/history-prompt-e.txt"
test_size = 100

top_k = 1
interval = 1
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open(api_key_path) as f:
    openai.api_key = f.read().strip()

with jsonlines.open(data_path) as f:
    con = [line for line in f]


def generate_headline_history_based_gpt3(rec, prompt):
    try:
        history = rec['history']
        article = rec['src']
        if len(article.split(' ')) > 2000:
            article = ' '.join(article.split(' ')[:2000])
        prompt_tp = prompt % (history, article)
        response = openai.Completion.create(model=model_name, prompt=prompt_tp, temperature=0, max_tokens=32)
        return response["choices"][0]['text'].strip().replace('\n', ' ')
    except:
        print('Server error, restart in 5s')
        time.sleep(5)
        return generate_headline_history_based_gpt3(rec, prompt)


def judge_time(previous):
    now = time.time()
    while now - previous < interval:
        time.sleep(1)
        now = time.time()
    return now


def update_result_file(output_path, idx, headline):
    with open(output_path, 'a') as f:
        f.write('%s\t%s\n' % (idx, headline))


def get_processed_ids(output_path):
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = f.readlines()
        ids_current = set([int(result.split('\t')[0]) for result in results])
    else:
        return set([])
    return ids_current


test_set = con[:test_size]
progress = tqdm(range(len(test_set)))
previous = time.time() - 3
processed_ids = get_processed_ids(output_path)
for idx, rec in enumerate(test_set):
    if idx in processed_ids:
        _ = progress.update(1)
        continue
    previous = judge_time(previous)
    headline = generate_headline_history_based_gpt3(rec, prompt_f)
    update_result_file(output_path, idx, headline)
    _ = progress.update(1)










# prompt_direct = "Generate a headline for the following article: %s" % article
# prompt_tp = "Generate a headline for the following article focusing on %s: %s" % (kps_focus, article)

