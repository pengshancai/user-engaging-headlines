import tarfile
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from pytorch_lightning import LightningDataModule
import nltk
import time
from utils.io_utils import loadFromPKL


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


class DatasetSum(Dataset):
    def __init__(self, tarfile_name, indices, index_to_path):
        """
            tarfile_name is the name of the tarfile saves the pickle files (read only)
            indices are ids for gigaword instance id
            index_to_path is a function that map instance id to a path in the tarfile
        """
        super().__init__()
        self.indices = indices
        self.tarfile_name = tarfile_name
        self.index_to_path = index_to_path

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path = self.index_to_path(self.indices[idx])
        tf = tarfile.open(self.tarfile_name, "r")
        f = tf.extractfile(path)
        data_i = pickle.load(f)
        f.close()
        tf.close()
        return {
            "input_ids": torch.LongTensor(data_i["input_ids"]),
            "labels": torch.LongTensor(data_i["labels"])
        }


class DatasetUser(Dataset):
    def __init__(self, user_embs):
        super().__init__()
        if type(user_embs) == np.ndarray:
            user_embs = torch.from_numpy(user_embs)
        if len(user_embs.shape) == 1:
            user_embs = torch.unsqueeze(user_embs, dim=0)
        self.user_embs = user_embs

    def __len__(self):
        return len(self.user_embs)

    def __getitem__(self, idx):
        return torch.unsqueeze(self.user_embs[idx], dim=0)
        # return self.user_embs[idx]


class DatasetRecSum(Dataset):
    def __init__(self, dataset_sum: DatasetSum, dataset_user: DatasetUser):
        super().__init__()
        self.dataset_sum = dataset_sum
        self.dataset_user = dataset_user
        self.num_user = len(self.dataset_user)
        assert self.num_user > 0

    def __len__(self):
        return len(self.dataset_sum) * len(self.dataset_user)

    def __getitem__(self, idx):
        idx_sum = idx//self.num_user
        idx_user = idx % self.num_user
        ret = self.dataset_sum[idx_sum]
        info = {
            'input_ids': ret["input_ids"],
            'attention_mask': torch.ones_like(ret["input_ids"]).long(),
            'labels': ret["labels"],
            'user_emb': self.dataset_user[idx_user]
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
    def __init__(self, args, data_collator, train_dataset, valid_dataset, sampler=None):
        super().__init__()
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.sampler = sampler

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return super().test_dataloader()


def gigaword_index_to_path(idx):
    paths = [idx[:3], idx[8:12], idx[12:14], idx[14:16], idx[-4:]]
    return "/".join(paths) + ".pkl"


def load_processed_dataset(processed_summ_data_file, processed_summ_list_file, processed_user_data_file, index_to_path=gigaword_index_to_path):
    st_time = time.time()
    # Loading Summarization Dataset
    indices = loadFromPKL(processed_summ_list_file)
    dataset_summ = DatasetSum(tarfile_name=processed_summ_data_file, indices=indices, index_to_path=index_to_path)
    # Loading User Dataset
    processed_user_dataset = loadFromPKL(processed_user_data_file)[0]
    dataset_user = DatasetUser(processed_user_dataset)
    # Building RecSum Dataset
    dataset_recsum = DatasetRecSum(dataset_summ, dataset_user)
    print('Total loading time:\t %f minutes' % (float(time.time()-st_time) / 60))
    """
    time1 = time.time()
    with open(processed_summ_data_file, 'rb') as f:
        processed_summ_dataset = pickle.load(f)
    with open(processed_user_data_file, 'rb') as f:
        processed_user_dataset = pickle.load(f)[0]
    input_ids = processed_summ_dataset['input_ids']
    labels = processed_summ_dataset['labels']
    dataset_summ = DatasetSum(input_ids, labels)
    dataset_user = DatasetUser(processed_user_dataset)
    dataset_recsum = DatasetRecSum(dataset_summ, dataset_user)
    time3 = time.time()
    print('Total loading time:\t %f minutes' % (float(time3-time1) / 60))
    """
    return dataset_recsum

