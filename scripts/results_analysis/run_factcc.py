import torch
from utils.factcc_utils import evaluate, reformat_data
from transformers import BertForSequenceClassification, BertTokenizer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

checkpoint = '/data/home/pengshancai/workspace/recsum_/dump/factcc/factcc-checkpoint/'
model = BertForSequenceClassification.from_pretrained(checkpoint)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)
version = '1.3.1'

"""
Original: 
    1.3.3 (Pure Users): 68.58 
    1.3.1 (Mixed Users): 68.36
"""
data_dir = '../recsum_/results/newsroom/nr-pt-3.0-%s.json' % version
data_dir_l = reformat_data(data_dir, gold_as_claim=True)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
None-KP: 
    1.3.3 (Pure Users): 62.51
    1.3.1 (Mixed Users): 62.62
"""
data_dir = '../recsum_/results/newsroom/nr-pt-3.0-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP: 
    1.3.3 (Pure Users): 60.04
    1.3.1 (Mixed Users): 60.72
"""
data_dir = '../recsum_/results/newsroom/nr-sl-2.0-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-naive: 
    1.3.3 (Pure Users): 60.58
    1.3.1 (Mixed Users): 61.82
"""
data_dir = '../recsum_/results/newsroom/nr-sl-naive-2.0-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-random: 
    1.3.3 (Pure Users): 61.72
    1.3.1 (Mixed Users): 62.29
"""
data_dir = '../recsum_/results/newsroom/nr-sl-random-2.0-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-2: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-2.0-top-2-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-naive-2: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-naive-2.0-top-2-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-random-2: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-random-2.0-top-2-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-3: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-2.0-top-3-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-naive-3: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-naive-2.0-top-3-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-random-3: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-random-2.0-top-3-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-4: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-2.0-top-4-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-naive-4: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-naive-2.0-top-4-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-random-4: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-sl-random-2.0-top-4-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))

"""
KP-gold: 
    1.3.3 (Pure Users): 
    1.3.1 (Mixed Users): 
"""
data_dir = '../recsum_/results/newsroom/nr-gold-%s.json' % version
data_dir_l = reformat_data(data_dir)
preds = evaluate(model, tokenizer, device, data_dir_l)
print('correct_rate:\t%s' % (100 * (sum(preds)/len(preds))))
