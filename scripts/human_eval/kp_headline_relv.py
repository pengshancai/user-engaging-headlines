import json
import jsonlines
import random

dataset = 'gigaword'
results_path = '../recsum_/results/%s/' % dataset
exp_names = [
    'none-kp',
    'gold-kp',
    'kp-early-ft-3',
    'kp-early-ft-4',
]

with jsonlines.open('../recsum_/data/gigaword/synthesized_user/test.json') as f:
    data = [line for line in f]

results_all = {}
for exp_name in exp_names:
    with open(results_path + exp_name + '.json') as f:
        results_all[exp_name] = [rec[2] for rec in json.load(f)]

random_ids = random.sample(list(range(len(data))), 20)
for i, idx in enumerate(random_ids):
    print('%s\t%s' % (idx, data[idx]['kp_user_selected']))
    for exp_name, results_exp in results_all.items():
        print('%s:\t%s' % (exp_name, results_exp[idx]))
    print('original:\t%s' % data[idx]['tgt'])
    print()
