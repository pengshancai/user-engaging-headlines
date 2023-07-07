#!/usr/bin/env python
# coding=utf-8
"""
Pre-training for fine-tuning
Use special token <u> as prefix
"""

import logging
import argparse
import os
import pickle
import nltk
from filelock import FileLock
from transformers import (
    MODEL_MAPPING,
    SchedulerType,
    DataCollatorForSeq2Seq,
)
from transformers.utils import is_offline_mode
from models.summarizer import load_summarizer
from models.general import SummarizerPreTrainNaive
from utils.data_utils import DatasetRecSumPTNaive, DatasetSumm, DataModuleRecSum
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import load_dataset

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=False,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default='',
        help="A prefix to add before every source text.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default='text',
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default='title',
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_steps_epoch", type=int, default=30000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    '''
    Recommender args
    '''
    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--enable_gpu",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--news_query_vector_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--user_query_vector_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--news_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--user_log_length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_padded_news_embedding",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--user_log_mask",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--recommender_type",
        type=str,
        default='bart-base-encoder',
    )
    parser.add_argument(
        "--recommender_path",
        type=str,
        default='',
    )
    '''
    Summarizer args
    '''

    '''
    Data args
    '''
    parser.add_argument("--train_file_summ", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file_summ", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--data_cache_path", type=str, required=True, help="cache file to store dataset")
    parser.add_argument("--cache_file_name", type=str, default='cache-%s')
    parser.add_argument("--data_file_user", type=str, default='')
    parser.add_argument("--world_user_emb_file", type=str, default='')
    parser.add_argument("--per_device_batch_size_mle", type=int, default=32)
    '''
    PL args
    '''
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--n_gpus", type=int, default=-1)
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_2")
    parser.add_argument("--valid_per_epoch", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--warmup_rates", type=float, default=0.05)
    parser.add_argument("--train_steps", type=float, default=0.05)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    args = parser.parse_args()
    return args


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    """
    Step 0: Preparation
    """
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    """
    Step 1: Load models
    """
    summarizer, tokenizer = load_summarizer(args)
    model = SummarizerPreTrainNaive(args, summarizer, tokenizer)

    """
    Step 2： Load datasets
    """
    data_files_summ = {}
    if args.train_file_summ is not None:
        data_files_summ["train"] = args.train_file_summ
    if args.validation_file_summ is not None:
        data_files_summ["validation"] = args.validation_file_summ
    extension = data_files_summ[list(data_files_summ.keys())[0]].split(".")[-1]
    # raw_datasets_summ = load_dataset(extension, data_files=data_files_summ, field='data')
    if extension == 'jsonl':
        extension = 'json'
    raw_datasets_summ = load_dataset(extension, data_files=data_files_summ)
    datasets_summ = {}
    for split in raw_datasets_summ:
        datasets_summ[split] = DatasetSumm(args, raw_datasets_summ[split], tokenizer, args.cache_file_name % split)
    train_dataset = DatasetRecSumPTNaive(datasets_summ['train'])
    valid_dataset = DatasetRecSumPTNaive(datasets_summ['validation'])

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    args.label_pad_token_id = label_pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True,
    )
    data_module = DataModuleRecSum(args, data_collator, train_dataset, valid_dataset)

    '''
    Step 3: Fit the model
    '''
    print('Start training')
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{valid_loss:.8f}',
        save_last=True,
        monitor=None,
        mode="min",
        save_top_k=-1,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.n_gpus,
        strategy=args.strategy,
        precision=16,
        max_epochs=args.num_train_epochs,
        callbacks=[checkpoint_callback],
        sync_batchnorm=True,
        val_check_interval=1.0 / args.valid_per_epoch,
        profiler="simple",
        gradient_clip_val=args.gradient_clip_val  # 0.1
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

