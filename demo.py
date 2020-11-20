import torch
import json
from transformers import BertTokenizer
from compare_mt.rouge.rouge_scorer import RougeScorer
import logging
from nltk.tokenize import sent_tokenize
import numpy as np
import sys
from model import Refactor

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)

MAX_LEN = 512
tok = BertTokenizer.from_pretrained('bert-base-uncased')
device = "cuda"
scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

def compute_rouge(ref, hyp):
    ref = sent_tokenize(ref)
    hyp = sent_tokenize(hyp)
    score = scorer.score("\n".join(ref), "\n".join(hyp))
    return score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure

def to_tensor(sent, max_len=MAX_LEN):
    sent = sent.split()
    sent = " ".join(sent)
    ids = tok.encode(sent, add_special_tokens=False)
    ids = [tok.cls_token_id] + ids[:max_len - 2] + [tok.sep_token_id]
    return torch.LongTensor(ids).unsqueeze(0)  # batch_size is 1

def padding(sents):
    max_length = max(x.size(1) for x in sents)
    out = torch.ones(len(sents), max_length, dtype=torch.long) * tok.pad_token_id
    for (i, x) in enumerate(sents):
        x = x[0]
        out[i, :x.size(0)] = x
    return out.unsqueeze(0)

def scoring(data_pt, model_pt, result_pt):
    model = Refactor('bert-base-uncased', num_layers=2).to(device)
    model.load_state_dict(torch.load(model_pt), map_location=device)
    model = model.eval()
    rouge1, rouge2, rougeLsum = 0, 0, 0
    num = 0
    with open(data_pt) as f, open(result_pt, "w") as out_f:
        for (i, x) in enumerate(f):
            if (i + 1) % 100 == 0:
                print(f"{i + 1} samples evaluated")
            x = json.loads(x)
            text_id = to_tensor(x["article"]).to(device)
            summary_id = to_tensor(x["summary"]).to(device)
            candidate_ids = [to_tensor(x) for x in x["candidates"]]
            candidate_ids = padding(candidate_ids).to(device)
            with torch.no_grad():
                output = model(text_id, candidate_ids, summary_id)
            similarity = output["score"].squeeze(0).cpu().numpy()
            max_id = similarity.argmax()
            selected = x["candidates"][max_id]
            print(selected, file=out_f)
            r1, r2, rL = compute_rouge(x["summary"], selected)
            rouge1 += r1
            rouge2 += r2
            rougeLsum += rL
            num += 1
    print("ROUGE score calculated by compare_mt:")
    print(f"rouge-1: {rouge1 / num * 100:.6f}, rouge-2: {rouge2 / num * 100:.6f}, rouge-L: {rougeLsum / num * 100:.6f}")

if __name__ == "__main__":
    scoring(sys.argv[1], sys.argv[2], sys.argv[3])
