"""
Preprocess the dataset and add <u> to each sequence
"""
import json
import pickle
import tarfile
import torch
from models import load_tokenizer_user_feature
from tqdm import tqdm
from io import BytesIO
# from utils.data_utils import DatasetSum, DatasetUser, DatasetRecSum
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

with open('../recsum_/za/args_summ/s-1.0.pkl', 'rb') as f:
    args = pickle.load(f)
    args.max_source_length = 512
    args.max_target_length = 64
    args.stage = 'pt'
    args.processed_summ_data_file = '../recsum_/data/gigaword/processed_summ-data-%s-%s-%s.tar'
    args.processed_summ_data_list = '../recsum_/data/gigaword/processed_summ-list-%s-%s-%s.pkl'
    args.processed_user_data_file = '../recsum_/data/gigaword/processed_user-%s-%s-filtered.pkl'

tokenizer = load_tokenizer_user_feature(args)
prefix = args.source_prefix
BATCH_SIZE = 100


def batch_tokenize(inputs, targets):
    assert len(inputs) == len(targets)
    num_batches = int(len(inputs) / BATCH_SIZE)
    input_ids = []
    labels = []
    progress = tqdm(range(num_batches + int(num_batches * BATCH_SIZE < len(inputs))))
    for i in range(num_batches):
        inputs_batch = inputs[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        targets_batch = targets[i*BATCH_SIZE: (i+1)*BATCH_SIZE]

        model_input_batch = tokenizer(inputs_batch, max_length=args.max_source_length, padding=False, truncation=True)
        input_ids_batch = model_input_batch['input_ids']
        with tokenizer.as_target_tokenizer():
            targets_batch = tokenizer(targets_batch, max_length=args.max_target_length, padding=False, truncation=True)
        labels_batch = targets_batch['input_ids']
        input_ids += input_ids_batch
        labels += labels_batch
        _ = progress.update(1)

    # Last Batch
    if num_batches * BATCH_SIZE < len(inputs):
        inputs_batch = inputs[num_batches*BATCH_SIZE: ]
        targets_batch = targets[num_batches*BATCH_SIZE: ]

        model_input_batch = tokenizer(inputs_batch, max_length=args.max_source_length, padding=False, truncation=True)
        input_ids_batch = model_input_batch['input_ids']
        with tokenizer.as_target_tokenizer():
            targets_batch = tokenizer(targets_batch, max_length=args.max_target_length, padding=False, truncation=True)
        labels_batch = targets_batch['input_ids']
        input_ids += input_ids_batch
        labels += labels_batch
        _ = progress.update(1)

    return input_ids, labels
    # sum_dataset = DatasetSum(input_ids, attention_mask, labels)
    # return sum_dataset


def get_overlap(title, txt):
    return len(set(title.lower().split()) & set(title.lower().split())) * 1.0 / len(txt.split())


def judge_good_title(title, txt):
    if len(title.split(' ')) < 6:
        return False
    if get_overlap(title, txt) < 0.55:
        return False
    return True


processed_summ_datasets = {}
for split in ['valid', 'test', 'train']:
    print('Processing %s set\n' % split)
    with open('../recsum_/data/gigaword/%s.jsonl' % split, 'r') as json_file:
        con = list(json_file)
    
    print('Reading content')
    indices = []
    inputs = []
    targets = []
    progress = tqdm(range(len(con)))
    for rec in con:
        rec = json.loads(rec)
        
        title = rec["headline"]
        doc = ' '.join(rec["text"])
    
        #if judge_good_title(title, doc):
        indices.append(rec["id"])
        inputs.append(doc)
        targets.append(title)
        _ = progress.update(1)
    print('Tokenizing..')
    inputs = [prefix + inp for inp in inputs]
    input_ids, labels = batch_tokenize(inputs, targets)
    print(len(inputs))
    print('Writting to TarFile')
    tarfile_name = args.processed_summ_data_file % (args.max_source_length, args.max_target_length, 'validation' if split == 'valid' else split)
    listfile_name = args.processed_summ_data_list % (args.max_source_length, args.max_target_length, 'validation' if split == 'valid' else split)

    split_indices = []
    tf = tarfile.open(tarfile_name, "w")
    progress = tqdm(range(len(labels)))
    for idx, input_id, label in zip(indices, input_ids, labels):
        paths = [idx[:3], idx[8:12], idx[12:14], idx[14:16], idx[-4:]]
        path = "/".join(paths) + ".pkl"

        # Write to BytesIO
        pf = BytesIO()
        pickle.dump({"index":idx, "input_ids": input_id, "labels": label}, pf)
        pf.seek(0)
        
        # Get TarFile.tarinfo
        info = tf.tarinfo(name=path)
        info.size = len(pf.getbuffer())

        # Write to TarFile
        tf.addfile(info, pf)
        split_indices.append(idx)
        _ = progress.update(1)

    tf.close()
    # Saving List of Indices
    with open(listfile_name, "wb") as f:
        pickle.dump(split_indices, f)

    """
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
    """

"""
for split in processed_summ_datasets:
    print(split)
    with open(args.processed_summ_data_file % (args.max_source_length, args.max_target_length, split), 'wb') as f:
        pickle.dump(processed_summ_datasets[split], f)
"""

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
