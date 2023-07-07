"""
This script generates key phrases from the
"""
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
import json
import torch
import argparse
from datasets import load_dataset
import math
from tqdm import tqdm
from typing import Union

DO_SAMPLE = True
device = torch.device('cuda')


def parse_args():
    parser = argparse.ArgumentParser(description="Generate key phrases from a summarization dataset")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)
    parser.add_argument("--extract_target", type=str, default='text', help="the source to extract key-phrases from, could be either 'text' or 'title'")
    parser.add_argument("--extract_split", type=str, default='dev', help=" could be either train / dev (valid in gigaword) / test")
    parser.add_argument("--src_max_length", type=int, default=512)
    parser.add_argument("--tgt_max_length", type=int, default=None)
    parser.add_argument("--tgt_min_length", type=int, default=None)
    parser.add_argument("--begin_percentage", type=int, default=0)
    parser.add_argument("--end_percentage", type=int, default=100)
    parser.add_argument("--input_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')
    parser.add_argument("--identifier_column", type=str, default='url', help="A specific identifier for a passage, (url for newsroom / id for gigaword)")
    parser.add_argument("--corpus", type=str, default='gigaword', help="newsroom / gigaword")
    parser.add_argument("--hg_model_name", type=str, default='ankur310794/bart-base-keyphrase-generation-kpTimes')
    # parser.add_argument("--version", type=str, help='identifier for the version', default='')
    args = parser.parse_args()
    return args


def process_text(args, txts):
    if args.corpus == 'newsroom':
        return txts
    elif args.corpus == 'gigaword':
        if args.extract_target == 'text':
            txts_ = []
            for txt in txts:
                txt_ = ''
                for para in txt:
                    txt_ += para + ' '
                txts_.append(txt_)
            return txts_
        else:
            assert args.extract_target == 'headline'
            txts_ = [txt.replace('///', '') for txt in txts]
            return txts_
    else:
        assert False


def generate_kps(args):
    tokenizer = BartTokenizer.from_pretrained(args.hg_model_name)
    model = BartForConditionalGeneration.from_pretrained(args.hg_model_name).to(device)
    model.eval()
    # base_path = '/data/home/pengshancai/workspace/recsum_/data/newsroom/'
    # base_path = '/data/home/pengshancai/workspace/recsum_/data/gigaword/'
    data_files = {'validation': args.input_path + '%s.jsonl' % args.extract_split}  # The original newsroom jsonl file
    raw_dataset = load_dataset('json', data_files=data_files, split=f"validation[{args.begin_percentage}%:{args.end_percentage}%]")
    num_batches = math.ceil(len(raw_dataset) / args.batch_size)
    batch_start_ids = [i*args.batch_size for i in range(num_batches)]
    idx2kps = {}
    progress = tqdm(range(num_batches), desc='Generating key phrases')
    for start_id in batch_start_ids:
        input_text = raw_dataset[args.extract_target][start_id: start_id+args.batch_size]
        input_text = process_text(args, input_text)
        inputs = tokenizer(input_text, max_length=args.src_max_length, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, min_length=args.tgt_min_length, max_length=args.tgt_max_length, do_sample=DO_SAMPLE)
        kps_batch = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
        ids = raw_dataset[args.identifier_column][start_id: start_id+args.batch_size]
        for kps, idx in zip(kps_batch, ids):
            idx2kps[idx] = kps
        _ = progress.update(1)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(args.output_path):
    #     os.mkdir(args.output_path)
    with open(args.output_path + '%s-id2%skps-%s-%s.json' % (args.extract_split, args.extract_target, args.begin_percentage, args.end_percentage), 'w') as f:
        json.dump(idx2kps, f)


if __name__ == "__main__":
    args = parse_args()
    # with open('../recsum_/za/args/args.pkl', 'wb') as f:
    #     pickle.dump(args, f)
    generate_kps(args)




