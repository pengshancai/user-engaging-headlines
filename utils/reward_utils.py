import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from torch import nn

MAX_LEN_RETRIEVER = 512
MAX_LEN_RECOMMENDER = 24


class RecommenderScorer(nn.Module):
    def __init__(self, recommender, tokenizer):
        super(RecommenderScorer, self).__init__()
        self.recommender = recommender
        self.tokenizer = tokenizer

    def get_score(self, titles, user_embs, device):
        assert user_embs.shape[0] == len(titles)
        titles_encoded = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN_RECOMMENDER)
        with torch.no_grad():
            titles_ids, titles_mask = titles_encoded['input_ids'].to(device), titles_encoded['attention_mask'].to(device)
            input_ids = torch.cat((titles_ids, titles_mask), dim=1)  # Required format of the recommendation model
            titles_embs = self.recommender.get_news_emb(torch.unsqueeze(input_ids, dim=0)).squeeze(0)
            scores = torch.sum(titles_embs * user_embs.squeeze(1), dim=1)
        return scores

    # def get_score_user_history(self, titles, user_history, device):
    #     """
    #     Args:
    #         titles: [title1, title2, ...]
    #         user_history: [history1, history2, ...]
    #         device: torch device
    #     Returns:
    #         score
    #     """
    #     titles_encoded = self.tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_RECOMMENDER)
    #     user_history_ids = self.tokenizer(user_history, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_RECOMMENDER)
    #     with torch.no_grad():
    #         titles_ids, titles_mask = titles_encoded['input_ids'].to(device), titles_encoded['attention_mask'].to(device)
    #         input_ids = torch.cat((titles_ids, titles_mask), dim=1)  # Required format of the recommendation model
    #         titles_embs = self.recommender.get_news_emb(torch.unsqueeze(input_ids, dim=0)).squeeze(0)



class RetrievalScorer(nn.Module):
    def __init__(self, retriever, tokenizer):
        super(RetrievalScorer, self).__init__()
        self.retriever = retriever
        self.tokenizer = tokenizer

    def get_score(self, titles, passages, device):
        titles_ids = self.tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_RETRIEVER)["input_ids"].to(device)
        passages_ids = self.tokenizer(passages, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN_RETRIEVER)["input_ids"].to(device)
        with torch.no_grad():
            titles_embs = self.retriever(titles_ids).pooler_output
            passages_embs = self.retriever(passages_ids).pooler_output
            assert len(titles_embs) == len(passages_embs)
            scores = torch.sum(titles_embs * passages_embs, dim=1)
        return scores


def load_retriever(retriever_name_or_path=None):
    if not retriever_name_or_path:
        retriever_name_or_path = "facebook/dpr-ctx_encoder-single-nq-base"
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(retriever_name_or_path)
    retriever = DPRContextEncoder.from_pretrained(retriever_name_or_path)
    return retriever, tokenizer






