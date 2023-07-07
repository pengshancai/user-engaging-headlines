"""
Preprocess the dataset by adding noisy key phrase prompt
k_1.1: Unlike 1.0, 1.1 only use one KP in generation
"""
import os
import jsonlines
import json

base_path = '../recsum_/data/gigaword/'


def process_kps(kps):
    if kps.endswith(';'):
        kps = kps[:-1]
    kps = list(set(kps.split(';')))
    return kps


def process_title(title):
    if ' : People.com' in title:
        title = title.replace(' : People.com', '')
    if '- NYTimes.com' in title:
        title = title.replace(' - NYTimes.com', '')
    return title


def process_text(text):
    text = text.replace('\n', ' ')
    return text


pass
for split in ['train', 'valid']:
    os.makedirs(base_path + 'kp_1.1/', exist_ok=True)
    with open(base_path + 'kp_1.0/%s-id2headlinekps.json' % split) as f:
        id2title_kps = json.load(f)
    with jsonlines.open(base_path + '%s.jsonl' % split) as reader:
        with jsonlines.open(base_path + 'kp_1.1/%s.jsonl' % split, mode='w') as writer:
            for line in reader:
                idx = line['id']
                title = process_title(line['headline'])
                text = process_text(' '.join(line['text']))
                title_kp0 = process_kps(id2title_kps[idx])[0]
                text_ = title_kp0 + '</s> ' + text
                info = {'text': text_, 'title': title}
                _ = writer.write(info)


if __name__ == "__main__":
    for split in ['train', 'valid']:
        os.makedirs(base_path + 'kp_1.1/', exist_ok=True)
        with open(base_path + 'kp_1.0/%s-id2headlinekps.json' % split) as f:
            id2title_kps = json.load(f)
        with jsonlines.open(base_path + '%s.jsonl' % split) as reader:
            with jsonlines.open(base_path + 'kp_1.1/%s.jsonl' % split, mode='w') as writer:
                for line in reader:
                    idx = line['id']
                    title = process_title(line['headline'])
                    text = process_text(' '.join(line['text']))
                    title_kp0 = process_kps(id2title_kps[idx])[0]
                    text_ = title_kp0 + '</s> ' + text
                    info = {'text': text_, 'title': title}
                    _ = writer.write(info)

