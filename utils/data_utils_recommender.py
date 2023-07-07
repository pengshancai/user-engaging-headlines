import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Optional, Union
from pytorch_lightning import LightningDataModule
import nltk
from torch.utils.data import RandomSampler
from collections import defaultdict
import random


class DatasetRecommender(Dataset):
    def __init__(self, args, users, tokenizer, cache_file_name):
        super(DatasetRecommender, self).__init__()
        self.args = args
        self.args.max_tgt_length = 512
        self.args.max_src_length = 32
        self.users = users
        self.tokenizer = tokenizer
        self.set_up(cache_file_name)

    def __len__(self):
        return len(self.users)

    def convert_to_features(self, examples):
        padding = "max_length" if self.args.pad_to_max_length else False
        srcs = examples[self.args.src_column]
        tgts = examples[self.args.tgt_column]
        tokenized_data = self.tokenizer(tgts, max_length=self.args.max_tgt_length, padding=padding, truncation=True)
        tokenized_data_src = self.tokenizer(srcs, max_length=self.args.max_src_length, padding=padding, truncation=True)
        tokenized_data['src_input_ids'] = tokenized_data_src['input_ids']
        tokenized_data['src_attention_mask'] = tokenized_data_src['attention_mask']
        tokenized_data['src_token_type_ids'] = tokenized_data_src['token_type_ids']
        return tokenized_data

    def set_up(self, cache_file_name_):
        if not os.path.exists(self.args.data_cache_path):
            os.mkdir(self.args.data_cache_path)
        cache_file_name = self.args.data_cache_path + cache_file_name_ if cache_file_name_ else None
        desc = "Running tokenizer on DatasetRecommender. "
        if cache_file_name:
            desc += 'Saving cached file to %s.' % cache_file_name
        self.users = self.users.map(
            self.convert_to_features,
            batched=True,
            num_proc=self.args.preprocessing_num_workers,
            remove_columns=self.users.column_names,  # [history, kp]
            load_from_cache_file=not self.args.overwrite_cache,
            cache_file_name=cache_file_name,
            desc=desc,
        )

    def __getitem__(self, idx):
        return self.users[idx]


@dataclass
class DataCollatorForRecommender:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        src_input_ids = [feature["src_input_ids"] for feature in features] if "src_input_ids" in features[0].keys() else None
        # We have to pad the src_input_ids before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if src_input_ids is not None:
            max_src_length = max(len(l) for l in src_input_ids)
            if self.pad_to_multiple_of is not None:
                max_src_length = (
                        (max_src_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder_ids = [self.label_pad_token_id] * (max_src_length - len(feature["src_input_ids"]))
                remainder_mask = [0] * (max_src_length - len(feature["src_input_ids"]))
                remainder_type = [0] * (max_src_length - len(feature["src_input_ids"]))
                if isinstance(feature["src_input_ids"], list):
                    feature["src_input_ids"] = (feature["src_input_ids"] + remainder_ids if padding_side == "right" else remainder_ids + feature["src_input_ids"])
                    feature["src_attention_mask"] = (feature["src_attention_mask"] + remainder_mask if padding_side == "right" else remainder_mask + feature["src_attention_mask"])
                    feature["src_token_type_ids"] = (feature["src_token_type_ids"] + remainder_type if padding_side == "right" else remainder_type + feature["src_token_type_ids"])
                elif padding_side == "right":
                    feature["src_input_ids"] = np.concatenate([feature["src_input_ids"], remainder_ids]).astype(np.int64)
                    feature["src_attention_mask"] = np.concatenate([feature["src_attention_mask"], remainder_mask]).astype(np.int64)
                    feature["src_token_type_ids"] = np.concatenate([feature["src_token_type_ids"], remainder_type]).astype(np.int64)
                else:
                    feature["src_input_ids"] = np.concatenate([remainder_ids, feature["src_input_ids"]]).astype(np.int64)
                    feature["src_attention_mask"] = np.concatenate([remainder_mask, feature["src_attention_mask"]]).astype(np.int64)
                    feature["src_token_type_ids"] = np.concatenate([remainder_type, feature["src_token_type_ids"]]).astype(np.int64)
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        return features


class DataModuleRecommender(LightningDataModule):
    def __init__(self, args, data_collator, train_dataset, valid_dataset):
        super().__init__()
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        # sampler = RandomSampler(self.train_dataset, num_samples=self.args.num_train_steps_epoch)
        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.per_device_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.args.per_device_batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return super().test_dataloader()



