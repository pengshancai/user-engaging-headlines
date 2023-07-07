import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import html as _html
import itertools as _itertools
import random as _random
from collections import namedtuple as _namedtuple
import spacy as _spacy
from os import system as _system
import sentence_transformers

MAX_LEN_RECOMMENDER = 24


def get_ut_rel_score_dpr(kp_user, title, encoder_query, encoder_ctx, tokenizer_dpr, device):
    """
    User KP - Title Relevance
    Args:
        kp_user:
        title:
        encoder_query:
        encoder_ctx:
        tokenizer_dpr:
        device:
    Returns:
    """
    inputs_kp = tokenizer_dpr([kp_user], return_tensors='pt', max_length=32).to(device)
    inputs_title = tokenizer_dpr([title], return_tensors='pt', max_length=128).to(device)
    with torch.no_grad():
        emb_kp = encoder_query(**inputs_kp).pooler_output
        emb_title = encoder_ctx(**inputs_title).pooler_output
        score = torch.matmul(emb_kp, emb_title.T)
    return score.cpu().item()


def get_ut_rel_score_bm25(kp_user, title):
    # TODO
    pass


def get_user_emb(user_histories, recommender, tokenizer_rec, device):
    histories_encoded = tokenizer_rec(user_histories, return_tensors="pt", padding=True, truncation=True,
                                      max_length=MAX_LEN_RECOMMENDER)
    histories_ids, histories_mask = histories_encoded['input_ids'].to(device), histories_encoded['attention_mask'].to(
        device)
    with torch.no_grad():
        input_ids = torch.cat((histories_ids, histories_mask), dim=1)  # Required format of the recommendation model
        log_mask = torch.ones(input_ids.shape[0], dtype=int)
        user_features = torch.unsqueeze(input_ids.to(device), 0)
        log_mask = torch.unsqueeze(log_mask.to(device), 0)
        user_emb = recommender.get_user_emb(user_features, log_mask)
    return user_emb


def get_title_emb(title, recommender, tokenizer_rec, device):
    titles_encoded = tokenizer_rec([title], return_tensors="pt", padding=True, truncation=True,
                                   max_length=MAX_LEN_RECOMMENDER)
    titles_ids, titles_mask = titles_encoded['input_ids'].to(device), titles_encoded['attention_mask'].to(device)
    with torch.no_grad():
        input_ids = torch.cat((titles_ids, titles_mask), dim=1)  # Required format of the recommendation model
        titles_embs = recommender.get_news_emb(torch.unsqueeze(input_ids, dim=0)).squeeze(0)
    return titles_embs


def get_rcmd_score(history, title, recommender, tokenizer_rec, device):
    user_emb = get_user_emb(history, recommender, tokenizer_rec, device).cpu().numpy()[0]
    title_emb = get_title_emb(title, recommender, tokenizer_rec, device).cpu().numpy()[0]
    score = np.dot(user_emb, title_emb)
    return score


def get_tt_rel_score_dpr(title, text, encoder_query, encoder_ctx, tokenizer_dpr, device):
    """
    Title Text Relevance
    Args:
        title:
        text:
        encoder_query:
        encoder_ctx:
        tokenizer_dpr:
        device:

    Returns:
    """
    inputs_title = tokenizer_dpr([title], return_tensors='pt', max_length=128).to(device)
    inputs_text = tokenizer_dpr([text], return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        emb_title = encoder_query(**inputs_title).pooler_output
        emb_text = encoder_ctx(**inputs_text).pooler_output
        score = torch.matmul(emb_text, emb_title.T)
    return score.cpu().item()


def get_corpus_level_bleu(titles, texts):
    hyps = [word_tokenize(title) for title in titles]
    refs = [[word_tokenize(sent) for sent in sent_tokenize(text)] for text in texts]
    return corpus_bleu(refs, hyps)


def get_sentence_level_bleu(titles, texts):
    hyps = [word_tokenize(title) for title in titles]
    refs = [[word_tokenize(sent) for sent in sent_tokenize(text)] for text in texts]
    return corpus_bleu(refs, hyps)


def get_bm25_scores_for_docs(bm25, query, docs):
    """
    Calculate bm25 scores between query and a new set of docs
    """
    if type(query) == str:
        query = word_tokenize(query.lower())
    if type(docs[0]) == str:
        docs = [word_tokenize(doc) for doc in docs]
    score = np.zeros(len(docs))
    doc_freqs = []
    for doc in docs:
        frequencies = {}
        for word in doc:
            if word not in frequencies:
                frequencies[word] = 0
            frequencies[word] += 1
        doc_freqs.append(frequencies)
    doc_len = np.array([len(doc) for doc in docs])
    for q in query:
        q_freq = np.array([(doc.get(q) or 0) for doc in doc_freqs])
        score += (bm25.idf.get(q) or 0) * (q_freq * (bm25.k1 + 1) /
                                           (q_freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)))
    return score


def get_clm_perplexity(title, tokenizer, model, device):
    if title == '':
        title = 'Nothing'
    inputs = tokenizer(title + ' <|endoftext|>', return_tensors='pt', max_length=128).to(device)
    # if len(inputs['input_ids'][0]) == 1:
    #     inputs = tokenizer("Title: " + title, return_tensors='pt').to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs['input_ids'])['loss'].cpu().item()
    # if math.isnan(loss):
    #     assert False
    return math.exp(loss)


def get_mlm_perplexity(title, tokenizer, model, device):
    if title == '':
        title = 'None'
    inputs = tokenizer(title, return_tensors='pt')['input_ids'].to(device)
    repeated_inputs = inputs.repeat(inputs.size(-1) - 2, 1)
    mask = torch.ones(inputs.size(-1) - 1).diag(1)[:-2].to(device)
    masked_inputs = repeated_inputs.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeated_inputs.masked_fill(masked_inputs != tokenizer.mask_token_id, -100)
    with torch.no_grad():
        loss = model(masked_inputs, labels=labels).loss
    perplexity = np.exp(loss.cpu().item())
    return perplexity


def get_sbert_score(sent1, sent2, sbert):
    emb1 = sbert.encode([sent1], convert_to_tensor=True)
    emb2 = sbert.encode([sent2], convert_to_tensor=True)
    cos_score = sentence_transformers.util.cos_sim(emb1, emb2).cpu().item()
    return cos_score


class Fragments(object):
    Match = _namedtuple("Match", ("summary", "text", "length"))
    @classmethod
    def _load_model(cls):
        if not hasattr(cls, "_en"):
            try:
                cls._en = _spacy.load("en_core_web_sm")
            except:
                _system("python -m spacy download en_core_web_sm")
                cls._en = _spacy.load("en_core_web_sm")

    def __init__(self, summary, text, tokenize = True, case = False):
        self._load_model()
        self._tokens = tokenize
        self.summary = self._tokenize(summary) if tokenize else summary.split()
        self.text    = self._tokenize(text)    if tokenize else text.split()
        self._norm_summary = self._normalize(self.summary, case)
        self._norm_text    = self._normalize(self.text, case)
        self._match(self._norm_summary, self._norm_text)

    def _tokenize(self, text):
        """
        Tokenizes input using the fastest possible SpaCy configuration.
        This is optional, can be disabled in constructor.

        """
        return self._en(text, disable = ["tagger", "parser", "ner", "textcat"])

    def _normalize(self, tokens, case = False):
        """
        Lowercases and turns tokens into distinct words.
        """
        return [
            str(t).lower()
            if not case
            else str(t)
            for t in tokens
        ]

    def overlaps(self):
        """
        Return a list of Fragments.Match objects between summary and text.
        This is a list of named tuples of the form (summary, text, length):
            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment
        """
        return self._matches

    def strings(self, min_length = 0, raw = None, summary_base = True):
        """
        Return a list of explicit match strings between the summary and reference.
        Note that this will be in the same format as the strings are input. This is
        important to remember if tokenization is done manually. If tokenization is
        specified automatically on the raw strings, raw strings will automatically
        be returned rather than SpaCy tokenized sequences.
        Arguments:
            - min_length (int): filter out overlaps shorter than this (default = 0)
            - raw (bool): return raw input rather than stringified
                - (default = False if automatic tokenization, True otherwise)
            - summary_base (true): strings are based of summary text (default = True)
        Returns:
            - list of overlaps, where overlaps are strings or token sequences
        """
        # Compute the strings against the summary or the text?
        base = self.summary if summary_base else self.text
        # Generate strings, filtering out strings below the minimum length.
        strings = [
            base[i : i + length]
            for i, j, length
            in self.overlaps()
            if length > min_length
        ]
        # By default, we just return the tokenization being used.
        # But if they user wants a raw string, then we convert.
        # Mostly, this will be used along with spacy.
        if self._tokens and raw:
            for i, s in enumerate(strings):
                strings[i] = str(s)
        # Return the list of strings.
        return strings

    def coverage(self, summary_base = True):
        """
        Return the COVERAGE score of the summary and text.
        Arguments:
            - summary_base (bool): use summary as numerator (default = True)
        Returns:
            - decimal COVERAGE score within [0, 1]
        """
        numerator = sum(o.length for o in self.overlaps())
        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.reference)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def density(self, summary_base = True):
        """
        Return the DENSITY score of summary and text.
        Arguments:
            - summary_base (bool): use summary as numerator (default = True)
        Returns:
            - decimal DENSITY score within [0, ...]
        """
        numerator = sum(o.length ** 2 for o in self.overlaps())
        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.reference)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def compression(self, text_to_summary = True):
        """
        Return compression ratio between summary and text.
        Arguments:
            - text_to_summary (bool): compute text/summary ratio (default = True)
        Returns:
            - decimal compression score within [0, ...]
        """
        ratio = [len(self.text), len(self.summary)]
        try:
            if text_to_summary: return ratio[0] / ratio[1]
            else:               return ratio[1] / ratio[0]
        except ZeroDivisionError:
            return 0

    def _match(self, a, b):
        """
        Raw procedure for matching summary in text, described in paper.
        """
        self._matches = []
        a_start = b_start = 0
        while a_start < len(a):
            best_match = None
            best_match_length = 0
            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start
                    while a_end < len(a) and b_end < len(b) \
                            and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1
                    length = a_end - a_start
                    if length > best_match_length:
                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length
                    b_start = b_end
                else:
                    b_start += 1
            b_start = 0
            if best_match:
                if best_match_length > 0:
                    self._matches.append(best_match)
                a_start += best_match_length
            else:
                a_start += 1

    def _htmltokens(self, tokens):
        """score_originalCarefully process tokens to handle whitespace and HTML characters."""
        return [
            [
                _html.escape(t.text).replace("\n", "<br/>"),
                _html.escape(t.whitespace_).replace("\n", "<br/>")
            ]    for t in tokens
        ]

    def annotate(self, min_length = 0, text_truncation = None, novel_italics = False):
        """Used to annotate fragments for website visualization.Arguments:
            - min_length (int): minimum length overlap to count (default = 0)
            - text_truncation (int): tuncated text length (default = None)
            - novel_italics (bool): italicize novel words (default = True)
        Returns:
            - a tuple of strings: (summary HTML, text HTML)
        """
        start = """
            <u
            style="color: {color}; border-color: {color};"
            data-ref="{ref}" title="Length: {length}"
            >
        """.strip()
        end = """
            </u>
        """.strip()
        # Here we tokenize carefully to preserve sane-looking whitespace.
        # (This part does require text to use a SpaCy tokenization.)
        summary = self._htmltokens(self.summary)
        text = self._htmltokens(self.text)
        # Compute novel word set, if requested.
        if novel_italics:
            novel = set(self._norm_summary) - set(self._norm_text)
            for word_whitespace in summary:
                if word_whitespace[0].lower() in novel:
                    word_whitespace[0] = "<em>" + word_whitespace[0] + "</em>"
        # Truncate text, if requested.
        # Must be careful later on with this.
        if text_truncation is not None:
            text = text[:text_truncation]
        # March through overlaps, replacing tokens with HTML-tagged strings.
        colors = self._itercolors()
        for overlap in self.overlaps():
            # Skip overlaps that are too short.
            if overlap.length < min_length:
                continue
            # Reference ID for JavaScript highlighting.
            # This is random, but shared between corresponding fragments.
            ref = _random.randint(0, 1e10)
            color = next(colors)
            # Summary starting tag.
            summary[overlap.summary][0] = start.format(
                color = color,
                ref = ref,
                length = overlap.length,
            ) + summary[overlap.summary][0]
            # Text starting tag.
            text[overlap.text][0] = start.format(
                color = color,
                ref = ref,
                length = overlap.length,
            ) + text[overlap.text][0]
            # Summary ending tag.
            summary[overlap.summary + overlap.length - 1][0] += end
            # Text ending tag.
            text[overlap.text + overlap.length - 1][0] += end
        # Carefully join tokens and whitespace to reconstruct the string.
        summary = " ".join("".join("".join(tw) for tw in summary).split())
        text = " ".join("".join("".join(tw) for tw in text).split())
        # Return the tuple.
        return summary, text

    def _itercolors(self):
        # Endlessly cycle through these colors.
        return _itertools.cycle((
            "#393b79",
            "#5254a3",
            "#6b6ecf",
            "#9c9ede",
            "#637939",
            "#8ca252",
            "#b5cf6b",
            "#cedb9c",
            "#8c6d31",
            "#bd9e39",
            "#e7ba52",
            "#e7cb94",
            "#843c39",
            "#ad494a",
            "#d6616b",
            "#e7969c",
            "#7b4173",
            "#a55194",
            "#ce6dbd",
            "#de9ed6",

        ))


def min_max_normalize():
    pass







# def get_corpus_bleu(preds, golds):
#     hyps = [word_tokenize(pred.lower()) for pred in preds]
#     refs = [word_tokenize(gold.lower()) for gold in golds]
#     # refs = [[word_tokenize(sent) for sent in sent_tokenize(text)] for text in texts]
#     progress = tqdm(range(len(hyps)))
#     scores = []
#     for i, hyp in enumerate(hyps):
#         _ = progress.update(1)
#         score = sentence_bleu(refs, hyp)
#         scores.append(score)
#         if i % 100 == 0:
#             print(np.mean(scores))
#     return np.mean(scores)
    # return corpus_bleu(refs, hyps)