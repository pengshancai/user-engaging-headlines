import torch
import torch.nn.functional as F
import nltk
import pytorch_lightning as pl
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from undecorated import undecorated
from types import MethodType
import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def shift_tokens_left(input_ids: torch.Tensor):
    """
    Shift input ids one token to the left.
    """
    decoder_end_token_id = input_ids[:, 0]
    batch_size, seq_len = input_ids.shape
    shifted_input_ids = input_ids.new_zeros((batch_size, seq_len-1))
    shifted_input_ids[:, :] = input_ids[:, 1:].clone()
    # if pad_token_id is None:
    #     raise ValueError("self.model.config.pad_token_id has to be defined.")
    # # replace possible -100 values in labels by `pad_token_id`
    # shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class SummarizerBase(pl.LightningModule):
    def __init__(self, args, summarizer):
        super().__init__()
        self.args = args
        self.summarizer = summarizer

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.summarizer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.summarizer.named_parameters() if any(nd in n for nd in no_decay)],
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


class SummarizerPreTrain(SummarizerBase):
    def __init__(self, args, summarizer, tokenizer):
        super().__init__(args, summarizer)

    def training_step(self, batch_data, batch_idx):
        # batch_data: input_ids, attention_mask, labels, decoder_input_ids, user_emb
        outputs = self.summarizer.forward_user_feature(**batch_data)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        input_ids, attention_mask, labels, decoder_input_ids, user_emb = \
            batch_data['input_ids'], batch_data['attention_mask'], batch_data['labels'], batch_data[
                'decoder_input_ids'], batch_data['user_emb']
        norm = float((labels != self.args.pad).float().sum())
        outputs = self.summarizer.forward_user_feature(**batch_data)
        loss = outputs.loss
        self.log("valid_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        return float(loss) * float(norm), float(norm)

    def validation_epoch_end(self, batch_outputs):
        total_loss = 0
        total_norm = 0
        for loss, norm in batch_outputs:
            total_loss += loss
            total_norm += norm
        avg_loss = total_loss / total_norm
        self.log("valid_loss", avg_loss)


class SummarizerPreTrainNaive(SummarizerBase):
    def __init__(self, args, summarizer, tokenizer):
        super().__init__(args, summarizer)

    def training_step(self, batch_data, batch_idx):
        # batch_data: input_ids, attention_mask, labels, decoder_input_ids, user_emb
        outputs = self.summarizer.forward(**batch_data)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        # print(batch_data.keys())
        input_ids, attention_mask, labels = \
            batch_data['input_ids'], batch_data['attention_mask'], batch_data['labels']
        norm = float((labels != self.args.pad).float().sum())
        outputs = self.summarizer.forward(**batch_data)
        loss = outputs.loss
        self.log("valid_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        return float(loss) * float(norm), float(norm)

    def validation_epoch_end(self, batch_outputs):
        total_loss = 0
        total_norm = 0
        for loss, norm in batch_outputs:
            total_loss += loss
            total_norm += norm
        avg_loss = total_loss / total_norm
        self.log("valid_loss", avg_loss)


class SummarizerFineTune(SummarizerBase):
    def __init__(self, args, summarizer, recommender_scorer, retriever_scorer):
        super().__init__(args, summarizer)
        self.recommender_scorer = recommender_scorer
        self.retriever_scorer = retriever_scorer

    def split_batch(self, batch, batch_size_mle, batch_size_rl):
        batch_mle, batch_rl = {}, {}
        if batch_size_mle > 0:
            for key in ['input_ids', 'attention_mask', 'labels', 'world_user_emb', 'decoder_input_ids']:
                key_ = 'user_emb' if key == 'world_user_emb' else key
                batch_mle[key_] = batch[key][0:batch_size_mle]
        if batch_size_rl > 0:
            for key in ['input_ids', 'attention_mask', 'labels', 'user_emb', 'decoder_input_ids']:
                batch_rl[key] = batch[key][batch_size_mle:batch_size_mle + batch_size_rl]
        return batch_mle, batch_rl

    def get_reward_scores(self, titles, passages, user_embs):
        """
        :param titles:
        :param passages:
        :param user_embs:
        :return: scores
        Temporarily we are just using two rewards, i.e. recommendation & relevance, could add more later
        """
        device = user_embs.device
        if self.args.alpha_rl_rec > 0:
            scores_rec = self.recommender_scorer.get_score(titles, user_embs, device) * self.args.alpha_rl_rec
        else:
            scores_rec = torch.zeros(len(passages)).to(device)
        if self.args.alpha_rl_rel > 0:
            scores_rel = self.retriever_scorer.get_score(titles, passages, device) * self.args.alpha_rl_rel
        else:
            scores_rel = torch.zeros(len(passages)).to(device)
        scores = scores_rec + scores_rel
        return scores, (scores_rec, scores_rel)

    # DEPRECATED
    # def get_seq_log_probs(self, output_ids, logits, beam_indices):
    #     batch_size, seq_len = output_ids.shape
    #     mask = (output_ids != self.summarizer.tokenizer.pad_token_id).int()[:, 1:]
    #     probs = torch.zeros((batch_size, seq_len - 1))
    #     for idx_batch in range(batch_size):
    #         for idx_token in range(seq_len - 1):
    #             idx_vocab = output_ids[idx_batch][idx_token + 1]
    #             prob = F.softmax(logits[idx_token][beam_indices[idx_batch][idx_token].item()], dim=0)[idx_vocab]
    #             probs[idx_batch][idx_token] = prob
    #     log_probs = torch.log(probs).to(mask.device)
    #     mean_log_probs = torch.div(torch.sum(log_probs * mask, dim=1), torch.sum(mask, dim=1))
    #     return mean_log_probs

    def get_log_probs(self, encoder_outputs_comb, labels):
        if labels[0][0] == self.summarizer.tokenizer.eos_token_id:
            labels = shift_tokens_left(labels)
        labels_attention_mask = (labels != self.summarizer.tokenizer.pad_token_id).int()
        outputs = self.summarizer(encoder_outputs=encoder_outputs_comb, labels=labels)
        probs = torch.gather(torch.softmax(outputs['logits'], dim=2), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.div(torch.sum(torch.log(probs) * labels_attention_mask, dim=1), torch.sum(labels_attention_mask, dim=1))
        return log_probs

    def generate_greedy_sequence(self, encoder_outputs_comb):
        gen_kwargs_greedy = {
            "max_length": self.args.max_target_length,
            "num_beams": 1,
            "encoder_outputs": encoder_outputs_comb.copy(),
            "do_sample": False,
            "output_scores": True,
            # "return_dict_in_generate": True,
        }
        # gen_results_greedy = self.summarizer.generate(**gen_kwargs_greedy)
        # output_ids_greedy = gen_results_greedy['sequences']
        output_ids_greedy = self.summarizer.generate(**gen_kwargs_greedy)
        decoded_greedy = self.summarizer.tokenizer.batch_decode(output_ids_greedy, skip_special_tokens=True)
        return decoded_greedy

    def generate_sample_sequence(self, encoder_outputs_comb):
        gen_kwargs_sample = {
            "max_length": self.args.max_target_length,
            "num_beams": self.args.num_beams,
            "encoder_outputs": encoder_outputs_comb.copy(),
            "do_sample": True,
            "output_scores": True,
            # "return_dict_in_generate": True,
        }
        # gen_results_sample = self.summarizer.generate(**gen_kwargs_sample)  # Slow
        # output_ids_sample = gen_results_sample['sequences']
        # logits_sample = gen_results_sample['scores']
        # beam_indices_sample = gen_results_sample['beam_indices']
        # log_probs_sample = self.get_seq_log_probs(output_ids_sample, logits_sample, beam_indices_sample)
        output_ids_sample = self.summarizer.generate(**gen_kwargs_sample)  # Slow
        log_probs_sample = self.get_log_probs(encoder_outputs_comb, output_ids_sample)
        decoded_sample = self.summarizer.tokenizer.batch_decode(output_ids_sample, skip_special_tokens=True)
        return decoded_sample, log_probs_sample

    def train_step_rl(self, batch_rl):
        input_ids, attention_mask, labels, decoder_input_ids, user_embs = \
            batch_rl['input_ids'], batch_rl['attention_mask'], batch_rl['labels'], batch_rl['decoder_input_ids'], \
            batch_rl['user_emb']
        encoder_outputs_comb = self.summarizer.get_encoder_output_user_feature(input_ids, attention_mask, user_embs)
        decoded_passages = self.summarizer.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_greedy = self.generate_greedy_sequence(encoder_outputs_comb)
        decoded_sample, log_probs_sample = self.generate_sample_sequence(encoder_outputs_comb)
        scores_greedy, (scores_greedy_rec, scores_greedy_rel) = self.get_reward_scores(decoded_greedy, decoded_passages,
                                                                                       user_embs)
        scores_sample, (scores_sample_rec, scores_sample_rel) = self.get_reward_scores(decoded_sample, decoded_passages,
                                                                                       user_embs)
        scores_diff_rec, scores_diff_rel = scores_sample_rec - scores_greedy_rec, scores_sample_rel - scores_greedy_rel
        # record the score difference of selected instances
        self.log('s_rec', scores_sample_rec[0], on_step=True, on_epoch=True, sync_dist=True, logger=True,
                 prog_bar=True)
        self.log('s_rel', scores_sample_rel[0], on_step=True, on_epoch=True, sync_dist=True, logger=True,
                 prog_bar=True)
        self.log('sd_rec', scores_diff_rec[0], on_step=True, on_epoch=True, sync_dist=True, logger=True,
                 prog_bar=True)
        self.log('sd_rel', scores_diff_rel[0], on_step=True, on_epoch=True, sync_dist=True, logger=True,
                 prog_bar=True)
        rl_losses = -(scores_sample - scores_greedy) * log_probs_sample
        rl_loss = torch.sum(rl_losses)
        return rl_loss

    def train_step_mle(self, batch_mle):
        outputs = self.summarizer.forward_user_feature(**batch_mle)
        mle_loss = outputs.loss
        return mle_loss

    def training_step(self, batch, batch_idx):
        batch_mle, batch_rl = self.split_batch(batch, self.args.per_device_batch_size_mle, self.args.per_device_batch_size_rl)
        rl_loss = self.train_step_rl(batch_rl)
        mle_loss = self.train_step_mle(batch_mle)
        self.log("RL_loss", rl_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        self.log("MLE_loss", mle_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        loss = rl_loss * self.args.alpha_mle + mle_loss * self.args.alpha_rl
        return loss

    def validation_step(self, batch, batch_idx):
        _, batch_rl = self.split_batch(batch, 0, self.args.per_device_batch_size_rl)
        input_ids, attention_mask, labels, decoder_input_ids, user_embs = \
            batch_rl['input_ids'], batch_rl['attention_mask'], batch_rl['labels'], batch_rl['decoder_input_ids'], \
            batch_rl['user_emb']
        decoded_passages = self.summarizer.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        encoder_outputs_comb = self.summarizer.get_encoder_output_user_feature(input_ids, attention_mask, user_embs)
        with torch.no_grad():
            decoded_titles, _ = self.generate_sample_sequence(encoder_outputs_comb)
        scores, (scores_rec, scores_rel) = self.get_reward_scores(decoded_titles, decoded_passages, user_embs)
        return scores_rec, scores_rel, decoded_titles, decoded_passages

    def record_validation_results(self, all_scores_rec, all_scores_rel, titles, passages):
        with open(self.args.log_path, 'a') as f:
            f.write('*** New checkpoint reached ***\n')
            for i, (score_rec, score_rel, title, passage) in enumerate(zip(all_scores_rec, all_scores_rel, titles, passages)):
                f.write('# IDX %s\n' % i)
                f.write('\tPASSAGE:\t%s\n' % passage.replace('\n', ''))
                f.write('\tTITLE:\t%s\n' % title)
                f.write('\tSCORE REC:\t%s\tSCORE REL:\t%s\n' % (score_rec, score_rel))

    def validation_epoch_end(self, batch_outputs):
        all_scores_rec, all_scores_rel, titles, passages = [], [], [], []
        for scores_rec, scores_rel, titles_batch, passages_batch in batch_outputs:
            all_scores_rec += list(scores_rec.cpu().numpy())
            all_scores_rel += list(scores_rel.cpu().numpy())
            titles += titles_batch
            passages += passages_batch
        self.record_validation_results(all_scores_rec, all_scores_rel, titles, passages)





