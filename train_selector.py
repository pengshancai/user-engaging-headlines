"""
Train KP selector
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
)
from models.selector import load_selector, Selector
from utils.data_utils_recommender import DatasetRecommender, DataCollatorForRecommender, DataModuleRecommender
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import load_dataset

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a key phrase selector")
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
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_steps_epoch", type=int, default=10000)
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
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    # parser.add_argument(
    #     "--checkpointing_steps",
    #     type=str,
    #     default=None,
    #     help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    # )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    '''
    Model args
    '''
    parser.add_argument("--dpr_ctx_encoder_path", type=str, required=True)
    parser.add_argument("--dpr_question_encoder_path", type=str, required=True)
    '''
    Data args
    '''
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--data_cache_path", type=str, required=True, help="cache file to store dataset")
    parser.add_argument("--cache_file_name", type=str, default='cache-%s')
    parser.add_argument("--max_tgt_length", type=int, default=128)
    parser.add_argument("--max_src_length", type=int, default=16)
    parser.add_argument("--src_column", type=str, default='kp')
    parser.add_argument("--tgt_column", type=str, default='history')
    '''
    PL args
    '''
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--n_gpus", type=int, default=-1)
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_2")
    parser.add_argument("--valid_per_epoch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--warmup_rates", type=float, default=0.05)
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    args = parser.parse_args()
    return args


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
    encoder_src, encoder_tgt, tokenizer = load_selector(args)
    model = Selector(args, encoder_src, encoder_tgt, tokenizer)
    """
    Step 2： Load datasets
    """
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = data_files[list(data_files.keys())[0]].split(".")[-1]
    raw_dataset_train = load_dataset(extension, data_files=data_files, split='train')
    raw_dataset_valid = load_dataset(extension, data_files=data_files, split='validation')
    train_dataset = DatasetRecommender(args, raw_dataset_train, tokenizer, args.cache_file_name % 'train')
    valid_dataset = DatasetRecommender(args, raw_dataset_valid, tokenizer, args.cache_file_name % 'validation')
    label_pad_token_id = tokenizer.pad_token_id
    args.label_pad_token_id = label_pad_token_id
    data_collator = DataCollatorForRecommender(
        tokenizer,
        model=encoder_tgt,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True,
    )
    data_module = DataModuleRecommender(args, data_collator, train_dataset, valid_dataset)
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


