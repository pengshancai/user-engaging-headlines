import json

path_in = '/data/home/pengshancai/workspace/recsum_/data/newsroom/%s.jsonl'
path_out = '/data/home/pengshancai/workspace/recsum_/data/newsroom/%s-lm.json'


for dtype in ['train', 'dev']:  #'dev', 'test',
    con_out = []
    with open(path_in % dtype) as f:
        for line in f:
            line = json.loads(line)
            con_out.append({'text': line['title'] + ' <|endoftext|>'})
    with open(path_out % dtype, 'w') as f_out:
        info = {
            'version': 'lm-1.0',
            'data': con_out,
        }
        json.dump(info, f_out)






