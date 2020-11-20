import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch import Tensor
import math
from transformers import BertModel

BERT_EMBED_DIM = 768


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    
    return TotalLoss


class Refactor(nn.Module):
    
    def __init__(self, encoder, hidden_size=768, nhead=8, num_layers=3):
        """ 
        refactor model
        encoder: pre-trained BERT model name
        nhead: number of heads of the transformer
        num_layers: number of layers of the transformer
        """
        super(Refactor, self).__init__()
        
        self.hidden_size = hidden_size
        self.encoder = BertModel.from_pretrained(encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def compute_score(src_emb, tgt_emb, weights):
        """ 
        compute similarity matrix
        src_emb: [bz, src_len, emb_size]
        tgt_emb: [bz, tgt_len, emb_size]
        weights: [bz, src_len]
        """
        mat_sim = torch.matmul(src_emb, tgt_emb.transpose(1, 2))  # [bz, src_len, tgt_len]
        recall, _ = mat_sim.max(2) # [bz, src_len]
        recall = torch.mul(recall, weights)
        recall = recall.sum(1)
        prec, _ = mat_sim.max(1)
        prec = prec.mean(1)
        recall = recall + 1
        prec = prec + 1
        r = 2 * torch.mul(recall, prec)
        r = torch.mul(r, 1 / (recall + prec))
        return r
         
    def forward(self, text_id, candidate_id, summary_id):
        
        batch_size = text_id.size(0)
        
        # get document embedding
        input_mask = ~(text_id == 0)
        doc_emb = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        # compute weights
        input_mask = ~input_mask
        weights = self.transformer(doc_emb.transpose(0, 1), src_key_padding_mask=input_mask)  # [seq_len, bz, dm]
        global_encoding = weights[0] # [bz, dm]
        weights = weights.transpose(0, 1)  # [bz, seq_len, dm]
        weights = torch.matmul(doc_emb, global_encoding.unsqueeze(-1)).squeeze(-1)
        weights = F.softmax(weights / math.sqrt(self.hidden_size), dim=1)
        
        # get summary embedding
        input_mask = ~(summary_id == 0)
        summary_emb = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == 0)
        candidate_emd = self.encoder(candidate_id, attention_mask=input_mask)[0]

        # scoring
        doc_emb = F.normalize(doc_emb, dim=2)
        summary_emb = F.normalize(summary_emb, dim=2)
        summary_score = self.compute_score(doc_emb, summary_emb, weights)
        doc_emb = torch.repeat_interleave(doc_emb, candidate_num, dim=0)
        weights = torch.repeat_interleave(weights, candidate_num, dim=0)
        candidate_emd = F.normalize(candidate_emd, dim=2)
        score = self.compute_score(doc_emb, candidate_emd, weights)
        score = score.view(batch_size, candidate_num)
        
        return {'score': score, 'summary_score': summary_score}