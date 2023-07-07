import json
import jsonlines
from tqdm import tqdm
"""
A KP is compared to a single title
"""

for split in ['train']:  # 'dev'
    with open('../recsum_/data/newsroom/kp_1.0/%s-url2textkps.json' % split) as f:
        url2text_kps = json.load(f)
    with open('../recsum_/data/newsroom/%s.jsonl' % split) as f:
        lines = f.readlines()
    progress = tqdm(range(len(lines)), desc='Processing %s set' % split)
    data = []
    with jsonlines.open('../recsum_/data/newsroom/selector/%s-selector-3.0.json' % split, 'w') as f_out:
        for line in lines:
            _ = progress.update(1)
            try:
                info = json.loads(line)
                url = info['url']
                title = info['title']
                kps_text = list(set(url2text_kps[url][:-1].split(';')))
                for kp in kps_text:
                    _ = f_out.write({'kp': kp, 'history': title})
                    # data.append({'kp': kp, 'history': title})
            except:
                continue





