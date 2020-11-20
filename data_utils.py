from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch
from transformers import BertTokenizer
import random
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

def bert_pad(X, pad_token_id=0, max_len=-1):
    if max_len < 0:
        max_len = max(len(x) for x in X)
    result = []
    for x in X:
        if len(x) < max_len:
            x.extend([pad_token_id] * (max_len - len(x)))
        result.append(x)
    return torch.LongTensor(result)

def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class RefactoringDataset(Dataset):
    def __init__(self, fdir, model_type, maxlen=100, is_test=False, total_len=512, is_sorted=True, maxnum=-1):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            self.num = len(self.files)
        self.tok = BertTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = maxlen
        self.is_test = is_test
        self.pad_token_id = self.tok.pad_token_id
        self.total_len = total_len
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        self.sorted = is_sorted
        self.maxnum = maxnum

    def __len__(self):
        return self.num

    def bert_encode(self, x, max_len=-1):
        _ids = self.tok.encode(x, add_special_tokens=False)
        ids = [self.cls_token_id]
        if max_len > 0:
            ids.extend(_ids[:max_len - 2])
        else:
            ids.extend(_ids[:self.total_len - 2])
        ids.append(self.sep_token_id)
        return ids

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        article = data["article"]
        src_txt = " ".join(article)
        src_input_ids = self.bert_encode(src_txt)
        abstract = data["abstract"]
        tgt_input_ids = self.bert_encode(" ".join(abstract))
        if self.maxnum > 0:
            candidates = data["candidates"][:self.maxnum]
            data["candidates"] = candidates
        if self.sorted:
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            data["candidates"] = candidates
        cand_txt = [" ".join(x[0]) for x in candidates]
        candidate_ids = [self.bert_encode(x, self.maxlen) for x in cand_txt]
        scores = [x[1] for x in candidates]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            "candidate_ids": candidate_ids,
            "scores": scores
            }
        if self.is_test:
            result["data"] = data
        return result

def collate_mp(batch, pad_token_id, is_test=False):
    def bert_pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(len(x) for x in X)
        result = []
        for x in X:
            if len(x) < max_len:
                x.extend([pad_token_id] * (max_len - len(x)))
            result.append(x)
        return torch.LongTensor(result)

    src_input_ids = bert_pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = bert_pad([x["tgt_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [bert_pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)
    scores = torch.FloatTensor([x["scores"] for x in batch])
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids, 
        "tgt_input_ids": tgt_input_ids,
        "candidate_ids": candidate_ids,
        "scores": scores
        }
    if is_test:
        result["data"] = data
    return result


class RefactoringIDsDataset(Dataset):
    def __init__(self, fdir, model_type, maxlen=100, max_num=-1, is_test=False, total_len=512, is_sorted=True):
        """ data format: article, abstract, indices, scores, cand_sents """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            self.num = len(self.files)
        self.tok = BertTokenizer.from_pretrained(model_type)
        self.maxlen = maxlen
        self.max_num = max_num
        self.is_test = is_test
        self.pad_token_id = self.tok.pad_token_id
        self.total_len = total_len
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        self.sorted = is_sorted

    def __len__(self):
        return self.num

    def bert_encode(self, x, max_len=-1):
        _ids = self.tok.encode(x, add_special_tokens=False)
        ids = [self.cls_token_id]
        if max_len > 0:
            ids.extend(_ids[:max_len - 2])
        else:
            ids.extend(_ids[:self.total_len - 2])
        ids.append(self.sep_token_id)
        return ids

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        article = data["article"]
        src_txt = " ".join(article)
        src_input_ids = self.bert_encode(src_txt)
        abstract = data["abstract"]
        tgt_input_ids = self.bert_encode(" ".join(abstract))
        indices = data["indices"]
        cand_sents = data["cand_sents"]
        candidates = [[cand_sents[id] for id in ids] for ids in indices]
        candidates = list(zip(candidates, data["scores"]))
        if self.max_num > 0:
            idxs = np.random.choice(len(candidates), size=min(len(candidates), self.max_num), replace=False)
            idxs = np.sort(idxs)
            candidates = [candidates[i] for i in idxs]
        if not self.sorted:
            random.shuffle(candidates)
        data["candidates"]  = candidates
        cand_txt = [" ".join(x[0]) for x in candidates]
        candidate_ids = [self.bert_encode(x, self.maxlen) for x in cand_txt]
        scores = [x[1] for x in candidates]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            "candidate_ids": candidate_ids,
            "scores": scores
            }
        if self.is_test:
            result["data"] = data
        return result

def collate_mp_ids(batch, pad_token_id, is_test=False):
    def bert_pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(len(x) for x in X)
        result = []
        for x in X:
            if len(x) < max_len:
                x.extend([pad_token_id] * (max_len - len(x)))
            result.append(x)
        return torch.LongTensor(result)
    
    def mat_pad(X):
        max_len = max(x.size(0) for x in X)
        zeros = torch.zeros(len(X), max_len, X[0].size(1), dtype=torch.long)
        for (i, x) in enumerate(X):
            zeros[i, :x.size(0)] = x
        return zeros

    src_input_ids = bert_pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = bert_pad([x["tgt_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [bert_pad(x, max_len) for x in candidate_ids]
    candidate_ids = mat_pad(candidate_ids)
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids, 
        "tgt_input_ids": tgt_input_ids,
        "candidate_ids": candidate_ids  
        }
    if is_test:
        result["data"] = data
    return result       

