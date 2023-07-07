import json
import jsonlines
from tqdm import tqdm
from collections import defaultdict

split = 'valid'
with open('../recsum_/data/gigaword/selector/%s-selector-2.0.json' % split) as f:
    lines = f.readlines()

kp2title = defaultdict(set)
progress = tqdm(range(len(lines)), desc='Processing data')
for line in lines:
    _ = progress.update(1)
    info = json.loads(line)
    kp, history = info['kp'], info['history']
    for title in history.split('; '):
        kp2title[kp].add(title)

progress = tqdm(range(len(kp2title)))
with jsonlines.open('../recsum_/data/gigaword/selector/%s-selector-3.0.json' % split, 'w') as f:
    for kp, titles in kp2title.items():
        _ = progress.update(1)
        for title in list(titles):
            _ = f.write({
                'kp': kp,
                'history': title
            })



