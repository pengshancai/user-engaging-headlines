"""
Preprocess the dataset by adding <u> to each sequence
"""
import json
import pickle
import torch
from models.summarizer import load_tokenizer_user_feature
from tqdm import tqdm
# from utils.data_utils import DatasetSum, DatasetUser, DatasetRecSum

with open('../recsum_/za/args_summ/s-1.0.pkl', 'rb') as f:
    args = pickle.load(f)
    args.max_source_length = 512
    args.max_target_length = 64
    args.stage = 'pt'
    args.processed_summ_data_file = '../recsum_/data/newsroom/processed_summ-%s-%s-%s.pkl'
    args.processed_user_data_file = '../recsum_/data/gigaword/processed_user-%s-%s.pkl'

tokenizer = load_tokenizer_user_feature(args)
prefix = args.source_prefix
BATCH_SIZE = 100


def batch_tokenize(inputs, targets):
    assert len(inputs) == len(targets)
    num_batches = int(len(inputs) / BATCH_SIZE) + 1
    input_ids = []
    attention_mask = []
    labels = []
    progress = tqdm(range(num_batches))
    for i in range(num_batches):
        inputs_batch = inputs[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        targets_batch = targets[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        model_input_batch = tokenizer(inputs_batch, max_length=args.max_source_length, padding=False, truncation=True)
        input_ids_batch = model_input_batch['input_ids']
        # attention_mask_batch = model_input_batch['attention_mask']
        with tokenizer.as_target_tokenizer():
            targets_batch = tokenizer(targets_batch, max_length=args.max_target_length, padding=False, truncation=True)
        labels_batch = targets_batch['input_ids']
        input_ids += input_ids_batch
        # attention_mask += attention_mask_batch
        labels += labels_batch
        _ = progress.update(1)
    return input_ids, labels
    # sum_dataset = DatasetSum(input_ids, attention_mask, labels)
    # return sum_dataset


processed_summ_datasets = {}
for split in ['valid', 'test', 'train']:
    print('Processing %s set\n' % split)
    with open('../recsum_/data/gigaword/%s.jsonl' % split, 'r') as json_file:
        con = list(json_file)
    print('Reading content')
    inputs = []
    targets = []
    progress = tqdm(range(len(con)))
    for rec in con:
        rec = json.loads(rec)
        title = rec["headline"]
        doc = ' '.join(rec["text"])
        inputs.append(doc)
        targets.append(title)
        _ = progress.update(1)
    print('Tokenizing..')
    inputs = [prefix + inp for inp in inputs]
    input_ids, labels = batch_tokenize(inputs, targets)
    processed_summ_dataset = {
        'input_ids': input_ids,
        'labels': labels,
    }
    if split == 'valid':
        # processed_summ_datasets['validation'] = processed_summ_dataset
        with open(args.processed_summ_data_file % (args.max_source_length, args.max_target_length, 'validation'), 'wb') as f:
            pickle.dump(processed_summ_dataset, f)
    else:
        # processed_summ_datasets[split] = processed_summ_dataset
        with open(args.processed_summ_data_file % (args.max_source_length, args.max_target_length, split), 'wb') as f:
            pickle.dump(processed_summ_dataset, f)

for split in processed_summ_datasets:
    print(split)
    with open(args.processed_summ_data_file % (args.max_source_length, args.max_target_length, split), 'wb') as f:
        pickle.dump(processed_summ_datasets[split], f)


for split in ['valid', 'test', 'train']:
    world_user_emb = torch.load(args.world_user_emb_file,
                                map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if split == 'valid':
        # processed_user_datasets['validation'] = [torch.unsqueeze(world_user_emb, dim=0).cpu().numpy()]
        with open(args.processed_user_data_file % (args.stage, 'validation'), 'wb') as f:
            pickle.dump([world_user_emb.cpu().numpy()], f)
    else:
        with open(args.processed_user_data_file % (args.stage, split), 'wb') as f:
            pickle.dump([world_user_emb.cpu().numpy()], f)
        # processed_user_datasets[split] = [torch.unsqueeze(world_user_emb, dim=0).cpu().numpy()]





