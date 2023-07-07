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


def pad_to_fix_len(x, fix_length, padding_front=True, padding_value=0):
    if padding_front:
        pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
        mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
    else:
        pad_x = x[:fix_length] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
    return pad_x, np.array(mask)


def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def load_logs(log_path):
    with open(log_path) as f:
        con = f.readlines()
    logs = []
    for line in con:
        log = line.strip().split('\t')
        logs.append(log)
    del con
    return logs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


class DataSetNRTitle(Dataset):
    def __init__(self, news_index, titles, logs, npratio, user_log_length, device):
        super().__init__()
        self.news_index = news_index
        self.titles = titles
        self.logs = logs
        self.npratio = npratio
        self.user_log_length = user_log_length
        self.device = device

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        return self.process_idx(idx)

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def gather_news(self, ids_doc):
        docs = []
        for idx in ids_doc:
            docs.append(self.titles[idx])
        return np.array(docs)

    def process_idx(self, idx):
        log = self.logs[idx]
        news_clicked = log[3].split()
        news_clicked, log_mask = pad_to_fix_len(self.trans_to_nindex(news_clicked), self.user_log_length)
        news_all = [i.split('-') for i in log[4].split()]
        news_neg = self.trans_to_nindex([i[0] for i in news_all if i[-1] == '0'])
        # Sample a portion of negative news from each user, if there exists negative news
        if len(news_neg) > 0:
            index_neg = news_sample(list(range(len(news_neg))), self.npratio)
            sam_neg = [news_neg[i] for i in index_neg]
        # Otherwise, use 0
        else:
            sam_neg = [0] * self.npratio
        news_pos = self.trans_to_nindex([i[0] for i in news_all if i[-1] == '1'])
        # Sample only one positive news from each user
        index_pos = news_sample(list(range(len(news_pos))), 1)
        sam_pos = [news_pos[i] for i in index_pos]
        sam_news = sam_pos + sam_neg
        news_feature = self.gather_news(sam_news)
        user_feature = self.gather_news(news_clicked)
        label = 0
        return user_feature, log_mask, news_feature, label

    def process_idx_dev(self, idx):
        line = self.logs[idx]
        click_docs = line[3].split()
        click_docs, log_mask = pad_to_fix_len(self.trans_to_nindex(click_docs), self.user_log_length)
        news_all = [i.split('-') for i in line[4].split()]
        news_neg = self.trans_to_nindex([i[0] for i in news_all if i[-1] == '0'])
        news_pos = self.trans_to_nindex([i[0] for i in news_all if i[-1] == '1'])
        news = news_pos + news_neg
        news_features = self.gather_news(news)
        user_features = self.gather_news(click_docs)
        labels = [1 for _ in range(len(news_pos))] + [0 for _ in range(len(news_neg))]
        # device convertion
        user_features = torch.unsqueeze(torch.LongTensor(np.array(user_features)).to(self.device), 0)
        log_mask = torch.unsqueeze(torch.FloatTensor(np.array(log_mask)).to(self.device), 0)
        news_features = torch.unsqueeze(torch.LongTensor(np.array(news_features)).to(self.device), 0)
        labels = torch.unsqueeze(torch.LongTensor(np.array(labels)).to(self.device), 0)
        return user_features, log_mask, news_features, labels

    def get_user_logs(self, idx):
        line = self.logs[idx]
        click_docs = line[3].split()
        click_docs, log_mask = pad_to_fix_len(self.trans_to_nindex(click_docs), self.user_log_length)
        user_features = self.gather_news(click_docs)
        user_features = torch.unsqueeze(torch.LongTensor(np.array(user_features)).to(self.device), 0)
        log_mask = torch.unsqueeze(torch.FloatTensor(np.array(log_mask)).to(self.device), 0)
        return user_features, log_mask


class DatasetSumm(Dataset):
    def __init__(self, args, raw_dataset, tokenizer, cache_file_name_=None):
        self.args = args
        self.data = raw_dataset
        self.tokenizer = tokenizer
        self.set_up(cache_file_name_)

    def set_up(self, cache_file_name_):
        cache_file_name = self.args.data_cache_path + cache_file_name_ if cache_file_name_ else None
        desc = "Running tokenizer on dataset. "
        if cache_file_name:
            desc += 'Saving cached file to %s.' % cache_file_name
        self.data = self.data.map(
            self.convert_to_features,
            batched=True,
            num_proc=self.args.preprocessing_num_workers,
            remove_columns=self.data.column_names,  # ['title', 'text']
            load_from_cache_file=not self.args.overwrite_cache,
            cache_file_name=cache_file_name,
            desc=desc,
        )

    def convert_to_features(self, examples):
        padding = "max_length" if self.args.pad_to_max_length else False
        inputs = examples[self.args.text_column]
        if type(inputs[0]) == list:
            # The text in the original gigaword dataset is put in the list, need convert to str
            inputs_ = []
            for input_list in inputs:
                input_str = ' '.join(input_list)
                inputs_.append(input_str)
            inputs = inputs_
        targets = examples[self.args.summary_column]
        inputs = [self.args.source_prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.args.max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and self.args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]['input_ids']), torch.LongTensor(self.data[idx]['labels'])


class DatasetUser(Dataset):
    def __init__(self, user_embs=None):
        super().__init__()
        if type(user_embs) is str:  # user_emb is a file path
            with open(user_embs, 'rb') as f:
                user_embs = pickle.load(f)
        if type(user_embs) == np.ndarray:
            user_embs = torch.from_numpy(user_embs)
        if len(user_embs.shape) == 1:
            user_embs = torch.unsqueeze(user_embs, dim=0)
        if len(user_embs.shape) == 2:
            user_embs = torch.unsqueeze(user_embs, dim=1)
        self.user_embs = user_embs

    def __len__(self):
        return len(self.user_embs)

    def __getitem__(self, idx):
        return self.user_embs[idx]


# class DatasetRecSum(Dataset):
#     def __init__(self, dataset_summ: DatasetSumm, dataset_user: DatasetUser):
#         super().__init__()
#         self.dataset_summ = dataset_summ
#         self.dataset_user = dataset_user
#         self.num_users = len(self.dataset_user)
#         assert self.num_users > 0
#
#     def __len__(self):
#         return len(self.dataset_summ) * len(self.dataset_user)
#
#     def __getitem__(self, idx):
#         idx_sum = idx // self.num_users
#         idx_user = idx % self.num_users
#         input_ids, labels = self.dataset_summ[idx_sum]
#         info = {
#             'input_ids': input_ids,
#             'attention_mask': torch.ones_like(input_ids).long(),
#             'labels': labels,
#             'user_emb': self.dataset_user[idx_user]
#         }
#         return info


class DatasetRecSumPT(Dataset):
    def __init__(self, dataset_summ: DatasetSumm, dataset_user: DatasetUser):
        super().__init__()
        self.dataset_summ = dataset_summ
        self.dataset_user = dataset_user
        self.num_users = len(self.dataset_user)
        assert self.num_users > 0

    def __len__(self):
        return len(self.dataset_summ) * len(self.dataset_user)

    def __getitem__(self, idx):
        idx_sum = idx // self.num_users
        idx_user = idx % self.num_users
        input_ids, labels = self.dataset_summ[idx_sum]
        info = {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids).long(),
            'labels': labels,
            'user_emb': self.dataset_user[idx_user]
        }
        return info


class DatasetRecSumPTNaive(Dataset):
    def __init__(self, dataset_summ: DatasetSumm):
        super().__init__()
        self.dataset_summ = dataset_summ

    def __len__(self):
        return len(self.dataset_summ)

    def __getitem__(self, idx):
        input_ids, labels = self.dataset_summ[idx]
        info = {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids).long(),
            'labels': labels,
        }
        return info


class DatasetRecSumFT(Dataset):
    def __init__(self,
                 dataset_summ: DatasetSumm,
                 dataset_user: DatasetUser,
                 dataset_world_user: DatasetUser,
                 num_users_per_summ=-1,
                 fixed_subset=False
                 ):
        super().__init__()
        self.dataset_summ = dataset_summ
        self.dataset_user = dataset_user
        self.dataset_world_user = dataset_world_user
        self.num_users = len(self.dataset_user)
        self.num_users_per_summ = num_users_per_summ
        assert self.num_users > 0
        if fixed_subset > 0:
            self.generate_fixed_subset()
        else:
            self.generate_random_subset()

    def generate_random_subset(self):
        print('Generating random subset')
        self.idx_sum2idx_user = defaultdict(list)
        ids_user = list(range(len(self.dataset_user)))
        for idx_sum in range(len(self.dataset_summ)):
            self.idx_sum2idx_user[idx_sum] = random.sample(ids_user, self.num_users_per_summ)

    def generate_fixed_subset(self):
        '''
        :param subset_size:
        :return: subset_ids
        Based on a rule, select a fixed subset of subset_size for validation, so that the validation set for the same subset size would also be the same
        '''
        print('Generating fixed subset')
        self.idx_sum2idx_user = defaultdict(list)
        for idx_sum in range(len(self.dataset_summ)):
            self.idx_sum2idx_user[idx_sum] = [idx_sum % self.num_users]
        self.num_users_per_summ = 1

    def __len__(self):
        return len(self.dataset_summ) * self.num_users_per_summ

    def __getitem__(self, idx):
        idx_sum = idx // self.num_users_per_summ
        idx_user = self.idx_sum2idx_user[idx_sum][idx % self.num_users_per_summ]
        input_ids, labels = self.dataset_summ[idx_sum]
        info = {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids).long(),
            'labels': labels,
            'user_emb': self.dataset_user[idx_user],
            'world_user_emb': self.dataset_world_user[0]
        }
        return info


@dataclass
class DataCollatorForRecSum:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        user_embs = [feature['user_emb'] for feature in features] if "user_emb" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                if 'user_emb' in feature:
                    _ = feature.pop('user_emb')
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model.summarizer, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.summarizer.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # prepare user_emb
        if len(user_embs[0].shape) == 2:
            user_embs = [torch.unsqueeze(user_emb, dim=0) for user_emb in user_embs]
        user_embs = torch.cat(user_embs, dim=0)
        features["user_emb"] = user_embs
        return features


class DataModuleRecSum(LightningDataModule):
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
            batch_size=self.args.per_device_batch_size_mle,
            # sampler=sampler,
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
            batch_size=self.args.per_device_batch_size_mle,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return super().test_dataloader()


