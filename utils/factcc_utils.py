from __future__ import absolute_import, division, print_function
import logging
import os
import json
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import numpy as np
import torch
from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']


task_name = 'factcc_annotated'
model_name_or_path = 'bert-base-uncased'
max_seq_length = 512
per_gpu_train_batch_size = 32
per_gpu_eval_batch_size = 32
model_type = 'bert'
# '../recsum_/results/newsroom/nr-sl-random-2.0-%s.json' % version


def reformat_data(result_dir, gold_as_claim=False):
    with open(result_dir) as f:
        recs = json.load(f)
    output_con = []
    for idx, rec in enumerate(recs):
        if gold_as_claim:
            src, claim, _ = rec
        else:
            src, _, claim = rec
        if src == '':
            src = 'No Source Input'
        if claim == '':
            claim = 'No Claim'
        info = {
            "id": idx,
            "text": src,
            "claim": claim
        }
        output_con.append(info)
    rfmt_dir = result_dir[:-5] + '-rfmt.json'
    with open(rfmt_dir, 'w') as outfile:
        json.dump(output_con, outfile)
    return rfmt_dir


def build_factcc_data(srcs, claims, save_path):
    print('Building FactCC dataset')
    output_con = []
    assert len(srcs) == len(claims)
    for idx, (src, claim) in enumerate(zip(srcs, claims)):
        if src == '':
            src = 'No Source Input'
        if claim == '':
            claim = 'No Claim'
        info = {
            "id": idx,
            "text": src,
            "claim": claim
        }
        output_con.append(info)
    with open(save_path, 'w') as outfile:
        json.dump(output_con, outfile)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None,
                 extraction_span=None, augmentation_span=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.extraction_span = extraction_span
        self.augmentation_span = augmentation_span


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a jsonl file."""
        with open(input_file) as f:
            lines = json.load(f)
        return lines
        # with open(input_file, "r", encoding="utf-8") as f:
        #     lines = []
        #     for line in f:
        #         lines.append(json.loads(line))
        # return lines


class FactCCManualProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(data_dir), "dev")

    def get_labels(self):
        """See base class."""
        return ["CORRECT", "INCORRECT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, example) in enumerate(lines):
            guid = str(i)
            if type(example) == str:
                example = example.replace("\'", "\"")
                example = json.loads(example)
            text_a = example["text"]
            text_b = example["claim"]
            # label = example["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


processors = {
    # "factcc_generated": FactCCGeneratedProcessor,
    "factcc_annotated": FactCCManualProcessor,
}

output_modes = {
    "factcc_generated": "classification",
    "factcc_annotated": "classification",
}


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 extraction_mask=None, extraction_start_ids=None, extraction_end_ids=None,
                 augmentation_mask=None, augmentation_start_ids=None, augmentation_end_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.extraction_mask = extraction_mask
        self.extraction_start_ids = extraction_start_ids
        self.extraction_end_ids = extraction_end_ids
        self.augmentation_mask = augmentation_mask
        self.augmentation_start_ids = augmentation_start_ids
        self.augmentation_end_ids = augmentation_end_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label : i for i, label in enumerate(label_list)}
    progress = tqdm(range(len(examples)))
    features = []
    for (ex_index, example) in enumerate(examples):
        _ = progress.update(1)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            if len(example.text_b) <= 3:
                example.text_b = 'nothing'
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        ####### AUX LOSS DATA
        # get tokens_a mask
        extraction_span_len = len(tokens_a) + 2
        extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]
        # get extraction labels
        if example.extraction_span:
            ext_start, ext_end = example.extraction_span
            extraction_start_ids = ext_start + 1
            extraction_end_ids = ext_end + 1
        else:
            extraction_start_ids = extraction_span_len
            extraction_end_ids = extraction_span_len
        augmentation_mask = [1 if extraction_span_len <= ix < extraction_span_len + len(tokens_b) + 1  else 0 for ix in range(max_seq_length)]
        if example.augmentation_span:
            aug_start, aug_end = example.augmentation_span
            augmentation_start_ids = extraction_span_len + aug_start
            augmentation_end_ids = extraction_span_len + aug_end
        else:
            last_sep_token = extraction_span_len + len(tokens_b)
            augmentation_start_ids = last_sep_token
            augmentation_end_ids = last_sep_token
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # if output_mode == "classification":
        #     label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     label_id = 0
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("ext mask: %s" % " ".join([str(x) for x in extraction_mask]))
            logger.info("ext start: %d" % extraction_start_ids)
            logger.info("ext end: %d" % extraction_end_ids)
            logger.info("aug mask: %s" % " ".join([str(x) for x in augmentation_mask]))
            logger.info("aug start: %d" % augmentation_start_ids)
            logger.info("aug end: %d" % augmentation_end_ids)
            logger.info("label: %d" % 0)
        extraction_start_ids = min(extraction_start_ids, 511)
        extraction_end_ids = min(extraction_end_ids, 511)
        augmentation_start_ids = min(augmentation_start_ids, 511)
        augmentation_end_ids = min(augmentation_end_ids, 511)
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=0,
                          extraction_mask=extraction_mask,
                          extraction_start_ids=extraction_start_ids,
                          extraction_end_ids=extraction_end_ids,
                          augmentation_mask=augmentation_mask,
                          augmentation_start_ids=augmentation_start_ids,
                          augmentation_end_ids=augmentation_end_ids))
    return features


def load_and_cache_examples(data_dir, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']),
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_ext_mask = torch.tensor([f.extraction_mask for f in features], dtype=torch.float)
    all_ext_start_ids = torch.tensor([f.extraction_start_ids for f in features], dtype=torch.long)
    all_ext_end_ids = torch.tensor([f.extraction_end_ids for f in features], dtype=torch.long)
    all_aug_mask = torch.tensor([f.augmentation_mask for f in features], dtype=torch.float)
    all_aug_start_ids = torch.tensor([f.augmentation_start_ids for f in features], dtype=torch.long)
    all_aug_end_ids = torch.tensor([f.augmentation_end_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                            all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                            all_aug_mask, all_aug_start_ids, all_aug_end_ids)
    return dataset


def make_model_input(batch):
    inputs = {'input_ids':        batch[0],
              'attention_mask':   batch[1],
              'token_type_ids':   batch[2],
              'labels':           batch[3]}
    return inputs


def complex_metric(preds, labels, prefix=""):
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "f1":   f1_score(y_true=labels, y_pred=preds, average="micro")
    }


def compute_metrics(task_name, preds, labels, prefix=""):
    assert len(preds) == len(labels)
    if task_name == "factcc_generated":
        return complex_metric(preds, labels, prefix)
    elif task_name == "factcc_annotated":
        return complex_metric(preds, labels, prefix)
    else:
        raise KeyError(task_name)


def evaluate(model, tokenizer, device, data_dir):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = task_name
    eval_dataset = load_and_cache_examples(data_dir, eval_task, tokenizer, evaluate=True)
    eval_batch_size = per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=eval_batch_size)
    # Eval!
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", eval_batch_size)
    # eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating using FactCC"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = make_model_input(batch)
            outputs = model(**inputs)
            # monitoring
            tmp_eval_loss = outputs[0]
            logits_ix = 1 if model_type == "bert" else 7
            logits = outputs[logits_ix]
            # eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    # result = compute_metrics(task_name, preds, out_label_ids)
    # eval_loss = eval_loss / nb_eval_steps
    # result["loss"] = eval_loss
    return preds





