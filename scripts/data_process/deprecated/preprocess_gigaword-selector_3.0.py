import json
import jsonlines
import random
from tqdm import tqdm

"""
A KP is compared to a single title
"""

idx2kps_file = '../recsum_/data/gigaword/kp_1.0/%s-id2textkps.json'
base_file = '../recsum_/data/gigaword/%s.jsonl'
output_file = '../recsum_/data/gigaword/selector/%s-selector-3.0.json'
SAMPLE_SIZE = 2
NUM_VALID_INSTANCE = 100000

for split in ['train']:  # 'train', 'valid', 'test'
    print('Processing %s set' % split)
    with open(idx2kps_file % split) as f:
        idx2text_kps = json.load(f)
    with open(base_file % split) as f:
        lines = f.readlines()
    data = []
    if split in {'valid', 'test'}:
        lines = lines[:NUM_VALID_INSTANCE]
    progress = tqdm(range(len(lines)))
    with jsonlines.open(output_file % split, 'w') as f:
        for line in lines:
            _ = progress.update(1)
            try:
                info = json.loads(line)
                idx = info['id']
                title = info['headline']
                kps_text = idx2text_kps[idx].split(';')[:4]
                kps_text_selected = random.sample(kps_text, SAMPLE_SIZE)
                for kp in kps_text_selected:
                    _ = f.write({'kp': kp, 'history': title})
            except:
                continue




