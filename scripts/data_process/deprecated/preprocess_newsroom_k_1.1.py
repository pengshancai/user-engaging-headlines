"""
Preprocess the dataset by adding noisy key phrase prompt
k_1.1: Key phrases are selected
"""
import json
import pickle
import torch.cuda
from models.general import SummarizerPreTrain
from utils.data_utils import DatasetSumm, DatasetUser, DataCollatorForRecSum, DatasetRecSumPT, DataModuleRecSum
from datasets import load_dataset
from models.summarizer import load_summarizer_naive


base_path = '../recsum_/data/newsroom/'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# extractor = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
# sent_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# sent_transformer.to(device)


def get_kps(text):
    pass


def process_title(title):
    if ' : People.com' in title:
        title = title.replace(' : People.com', '')
    if '- NYTimes.com' in title:
        title = title.replace(' - NYTimes.com', '')
    return title


def process_text(text):
    text = text.replace('\n', ' ')
    return text


for split in ['train', 'dev']:
    recs = []
    with open(base_path + '%s.jsonl' % split) as json_file:
        for line in json_file:
            info = json.loads(line)
            url = info['url']
            title = process_title(info['title'])
            text = process_text(info['text'])
            title_kps = get_kps(title)
    print('Writing files')
    with open('../recsum_/data/newsroom/%s-k_1.1.json' % split, 'w') as f:
        info_all = {
            'version': 'K-1.1',
            'data': recs
        }
        json.dump(info_all, f)


with open('../recsum_/dump/nr-ft-1.1/args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.source_prefix = ''

summarizer, tokenizer = load_summarizer_naive(args)
model = SummarizerPreTrain(args, summarizer, tokenizer)
data_files_summ = {}
data_files_summ["train"] = '../recsum_/data/newsroom/train-k_1.1.json'
data_files_summ["validation"] = '../recsum_/data/newsroom/dev-k_1.1.json'

extension = data_files_summ[list(data_files_summ.keys())[0]].split(".")[-1]
raw_datasets_summ = load_dataset(extension, data_files=data_files_summ, field='data')
datasets_summ = {}
for split in raw_datasets_summ:
    datasets_summ[split] = DatasetSumm(args, raw_datasets_summ[split], tokenizer, 'cache-%s-k_1.1' % split)

dataset_user = DatasetUser(args.data_file_user)
train_dataset = DatasetRecSumPT(datasets_summ['train'], dataset_user)
valid_dataset = DatasetRecSumPT(datasets_summ['validation'], dataset_user)

label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
args.label_pad_token_id = label_pad_token_id
data_collator = DataCollatorForRecSum(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=None,
    padding=True,
)
data_module = DataModuleRecSum(args, data_collator, train_dataset, valid_dataset)
