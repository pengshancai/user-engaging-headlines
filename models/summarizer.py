import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartEncoder,
    BartDecoder,
    BartForConditionalGeneration,
    shift_tokens_right,
)
from transformers import AutoTokenizer, AdamW
from recycle.recommender import NRModel
from typing import List, Optional, Tuple, Union
from transformers.generation_utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
)


def load_summarizer(args, ckpt_path=None, added_tokens=[]):
    if not hasattr(args, 'summarizer_model_path'):
        args.summarizer_model_path = args.model_name_or_path
    config = BartConfig.from_pretrained(args.summarizer_model_path, output_hidden_states=True)
    summarizer = BartForConditionalGeneration.from_pretrained(args.summarizer_model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.summarizer_model_path, use_fast=not args.use_slow_tokenizer)
    for token in added_tokens:
        tokenizer.add_tokens(token)  # <u>
    summarizer.resize_token_embeddings(len(tokenizer))
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['module']
        state_dict = {k.partition('module.summarizer.')[2]: state_dict[k] for k in state_dict.keys()}
        summarizer.load_state_dict(state_dict)
    summarizer.tokenizer = tokenizer
    return summarizer, tokenizer



# GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
# SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
# BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
# BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
#
# logger = logging.get_logger(__name__)
#
# BART_START_DOCSTRING = r"""
#     This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
#     library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
#     etc.)
#     This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
#     Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
#     and behavior.
#     Parameters:
#         config ([`BartConfig`]):
#             Model configuration class with all the parameters of the model. Initializing with a config file does not
#             load the weights associated with the model, only the configuration. Check out the
#             [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# """
#
# BART_INPUTS_DOCSTRING = r"""
#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.
#             Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.
#             [What are input IDs?](../glossary#input-ids)
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#             [What are attention masks?](../glossary#attention-mask)
#         decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
#             Indices of decoder input sequence tokens in the vocabulary.
#             Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.
#             [What are decoder input IDs?](../glossary#decoder-input-ids)
#             Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
#             is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
#             For translation and summarization training, `decoder_input_ids` should be provided. If no
#             `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
#             for denoising pre-training following the paper.
#         decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
#             Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
#             be used by default.
#             If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_inputs`] and
#             modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information
#             on the default strategy.
#         head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
#             Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
#             Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
#             Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,
#             1]`:
#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
#             Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
#             `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
#             hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
#             `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
#             Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
#             blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
#             If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
#             don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
#             `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape
#             `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you
#             can choose to directly pass an embedded representation. This is useful if you want more control over how to
#             convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
#         decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
#             Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
#             representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
#             input (see `past_key_values`). This is useful if you want more control over how to convert
#             `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
#             If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
#             of `inputs_embeds`.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# """
#
# BART_GENERATION_EXAMPLE = r"""
#     Summarization example:
#     ```python
#     >>> from transformers import BartTokenizer, BartForConditionalGeneration
#     >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
#     >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
#     >>> ARTICLE_TO_SUMMARIZE = (
#     ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
#     ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
#     ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
#     ... )
#     >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
#     >>> # Generate Summary
#     >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
#     >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
#     ```
#     Mask filling example:
#     ```python
#     >>> from transformers import BartTokenizer, BartForConditionalGeneration
#     >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
#     >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
#     >>> TXT = "My friends are <mask> but they eat too many carbs."
#     >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
#     >>> logits = model(input_ids).logits
#     >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
#     >>> probs = logits[0, masked_index].softmax(dim=0)
#     >>> values, predictions = probs.topk(5)
#     >>> tokenizer.decode(predictions).split()
#     ['not', 'good', 'healthy', 'great', 'very']
#     ```
# """
#
# _CHECKPOINT_FOR_DOC = "facebook/bart-base"
# _CONFIG_FOR_DOC = "BartConfig"
# _TOKENIZER_FOR_DOC = "BartTokenizer"
# # Base model docstring
# _EXPECTED_OUTPUT_SHAPE = [1, 8, 768]




# @add_start_docstrings(
#     "The BART Model which takes in src doc and user feature and output raw hidden-states",
#     BART_START_DOCSTRING,
# )
# class BartModelUserFeature(BartPretrainedModel):
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
#         self.encoder = BartEncoder(config, self.shared)
#         self.decoder = BartDecoder(config, self.shared)
#         # Initialize weights and apply final processing
#         self.post_init()
#         self.recommender = None
#
#     def set_recommender(self, recommender: NRModel):
#         # Load in and freeze the recommender
#         self.recommender = recommender
#         for name, param in self.recommender.named_parameters():
#             param.requires_grad = False
#
#     def get_input_embeddings(self):
#         return self.shared
#
#     def set_input_embeddings(self, value):
#         self.shared = value
#         self.encoder.embed_tokens = self.shared
#         self.decoder.embed_tokens = self.shared
#
#     def get_encoder(self):
#         return self.encoder
#
#     def get_decoder(self):
#         return self.decoder
#
#     @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         processor_class=_TOKENIZER_FOR_DOC,
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=Seq2SeqModelOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_EXPECTED_OUTPUT_SHAPE,
#     )
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, Seq2SeqModelOutput]:
#         # different to other models, Bart automatically creates decoder_input_ids from
#         # input_ids if no decoder_input_ids are provided
#         if decoder_input_ids is None and decoder_inputs_embeds is None:
#             if input_ids is None:
#                 raise ValueError(
#                     "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
#                     "passed, `input_ids` cannot be `None`. Please pass either "
#                     "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
#                 )
#             decoder_input_ids = shift_tokens_right(
#                 input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
#             )
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )
#         # user_emb = self.recommender.get_user_emb(user_log_ids, user_log_mask)  # batch_size, emb_dim
#         # user_emb_ = torch.unsqueeze(user_emb, dim=1)
#         # Concat encoder_outputs with user_emb
#         # encoder_outputs_comb = torch.cat((torch.unsqueeze(encoder_outputs[0][:, 0], 1), user_emb_, encoder_outputs[0][:, 2:]), dim=1)
#         # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs[0],
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         if not return_dict:
#             return decoder_outputs + encoder_outputs
#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
#
#     def forward_user_feature(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         user_emb: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, Seq2SeqModelOutput]:
#         # different to other models, Bart automatically creates decoder_input_ids from
#         # input_ids if no decoder_input_ids are provided
#         if decoder_input_ids is None and decoder_inputs_embeds is None:
#             if input_ids is None:
#                 raise ValueError(
#                     "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
#                     "passed, `input_ids` cannot be `None`. Please pass either "
#                     "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
#                 )
#             decoder_input_ids = shift_tokens_right(
#                 input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
#             )
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )
#         # Concat encoder_outputs with user_emb
#         encoder_outputs_comb = torch.cat((torch.unsqueeze(encoder_outputs[0][:, 0], 1), user_emb, encoder_outputs[0][:, 2:]), dim=1)
#         # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs_comb,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         if not return_dict:
#             return decoder_outputs + encoder_outputs
#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
#
#
# @add_start_docstrings(
#     "The BART Model with a language modeling head. Tailored for including user features. Can be used for summarization.", BART_START_DOCSTRING
# )
# class BartForConditionalGenerationUserFeature(BartPretrainedModel):
#     base_model_prefix = "model"
#     _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
#
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         self.model = BartModelUserFeature(config)
#         self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
#         self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
#
#         # Initialize weights and apply final processing
#         self.post_init()
#         self.recommender = None
#         self.world_user_emb = None
#         self.args = None
#
#     def set_recommender(self, recommender):
#         self.recommender = recommender
#
#     def set_args(self, args):
#         self.args = args
#
#     def get_encoder(self):
#         return self.model.get_encoder()
#
#     def get_decoder(self):
#         return self.model.get_decoder()
#
#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings
#
#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings
#
#     @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
#     @add_end_docstrings(BART_GENERATION_EXAMPLE)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, Seq2SeqLMOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
#         Returns:
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if labels is not None:
#             if use_cache:
#                 logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
#             use_cache = False
#             if decoder_input_ids is None and decoder_inputs_embeds is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )
#
#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
#
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#
#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )
#
#     def forward_user_feature(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[List[torch.FloatTensor]] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         user_emb: Optional[List[torch.FloatTensor]] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, Seq2SeqLMOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
#         Returns:
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         if labels is not None:
#             if use_cache:
#                 logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
#             use_cache = False
#             if decoder_input_ids is None and decoder_inputs_embeds is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )
#
#         outputs = self.model.forward_user_feature(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             # user_log_ids=user_log_ids,
#             # user_log_mask=user_log_mask,
#             user_emb=user_emb,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
#
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#
#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )
#
#     def get_encoder_output_user_feature(self, input_ids, attention_mask, user_emb):
#         encoder_outputs = self.get_encoder()(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         if len(user_emb.shape) == 2:
#             user_emb = torch.unsqueeze(user_emb, dim=1)
#         # Concat encoder_outputs with user_emb
#         encoder_outputs_comb = encoder_outputs.copy()
#         encoder_outputs_comb.last_hidden_state = torch.cat(
#             (torch.unsqueeze(encoder_outputs[0][:, 0], 1), user_emb, encoder_outputs[0][:, 2:]), dim=1)
#         return encoder_outputs_comb
#
#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         past=None,
#         attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         use_cache=None,
#         encoder_outputs=None,
#         **kwargs
#     ):
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]
#
#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "decoder_head_mask": decoder_head_mask,
#             "cross_attn_head_mask": cross_attn_head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }
#
#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
#
#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past
#
#
# def load_summarizer_user_feature(args, ckpt_path=None, added_tokens=[]):
#     if not hasattr(args, 'summarizer_model_path'):
#         args.summarizer_model_path = args.model_name_or_path
#     config = BartConfig.from_pretrained(args.summarizer_model_path, output_hidden_states=True)
#     summarizer = BartForConditionalGenerationUserFeature.from_pretrained(args.summarizer_model_path, config=config)
#     tokenizer = AutoTokenizer.from_pretrained(args.summarizer_model_path, use_fast=not args.use_slow_tokenizer)
#     for token in added_tokens:
#         tokenizer.add_tokens(token)  # <u>
#     summarizer.resize_token_embeddings(len(tokenizer))
#     summarizer.set_args(args)
#     if ckpt_path:
#         ckpt = torch.load(ckpt_path)
#         state_dict = ckpt['module']
#         state_dict = {k.partition('module.summarizer.')[2]: state_dict[k] for k in state_dict.keys()}
#         summarizer.load_state_dict(state_dict)
#     summarizer.tokenizer = tokenizer
#     return summarizer, tokenizer






