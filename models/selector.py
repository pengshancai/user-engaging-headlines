import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
import pytorch_lightning as pl
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Tuple, List
import numpy as np
import math

MAX_LEN_RETRIEVER = 512
TEMPERATURE_FACTOR = 0.07


def load_selector(args):
    if not args.dpr_ctx_encoder_path:
        args.dpr_ctx_encoder_path = "facebook/dpr-ctx_encoder-single-nq-base"
    if not args.dpr_question_encoder_path:
        args.dpr_question_encoder_path = "facebook/dpr-question_encoder-single-nq-base"
    encoder_src = DPRQuestionEncoder.from_pretrained(args.dpr_question_encoder_path)
    encoder_tgt = DPRContextEncoder.from_pretrained(args.dpr_ctx_encoder_path)
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.dpr_ctx_encoder_path)
    return encoder_src, encoder_tgt, tokenizer


class BiEncoderXEntLoss(object):
    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def calc(self,
        src_vectors: T,
        tgt_vectors: T
    ):
        logits = torch.matmul(src_vectors, tgt_vectors.T) * math.exp(TEMPERATURE_FACTOR)
        labels = torch.arange(src_vectors.size(0)).to(logits.device)
        loss = self.criterion(logits, labels)
        return loss


class Selector(pl.LightningModule):
    def __init__(self, args, encoder_src, encoder_tgt, tokenizer):
        super().__init__()
        self.args = args
        self.encoder_src = encoder_src
        self.encoder_tgt = encoder_tgt
        self.tokenizer = tokenizer
        self.loss_func = BiEncoderXEntLoss()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.encoder_tgt.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.encoder_tgt.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.encoder_src.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.encoder_src.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.num_train_steps_epoch * self.args.num_train_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch_data, batch_idx):
        src_vectors = self.encoder_src(
            input_ids=batch_data.src_input_ids,
            attention_mask=batch_data.src_attention_mask,
            token_type_ids=batch_data.src_token_type_ids
        )['pooler_output']
        tgt_vectors = self.encoder_tgt(
            input_ids=batch_data.input_ids,
            attention_mask=batch_data.attention_mask,
            token_type_ids=batch_data.token_type_ids
        )['pooler_output']
        loss = self.loss_func.calc(src_vectors, tgt_vectors)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        src_vectors = self.encoder_src(
            input_ids=batch_data.src_input_ids,
            attention_mask=batch_data.src_attention_mask,
            token_type_ids=batch_data.src_token_type_ids
        )['pooler_output']
        tgt_vectors = self.encoder_tgt(
            input_ids=batch_data.input_ids,
            attention_mask=batch_data.attention_mask,
            token_type_ids=batch_data.token_type_ids
        )['pooler_output']
        loss = self.loss_func.calc(src_vectors, tgt_vectors)
        return loss

    def validation_epoch_end(self, batch_outputs):
        total_loss = 0
        for loss in batch_outputs:
            total_loss += loss
        avg_loss = total_loss / len(batch_outputs)
        self.log("valid_loss", avg_loss)




