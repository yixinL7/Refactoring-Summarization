import json
from compare_mt.rouge.rouge_scorer import RougeScorer
from multiprocessing import Pool
import os
import random
from itertools import combinations
from functools import partial
import re
import nltk
import numpy as np

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

systems = ["match", "bart"]
scorer = RougeScorer(['rouge1'], use_stemmer=True)
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
# system order: match, bart, guided
# cnt = 0
# with open("./collection/score_bartandsum_val.jsonl") as f:
#     for (i, x) in enumerate(f):
#         x = json.loads(x)
#         if x["src"] is None:
#             continue
#         data = {"article": x["src"], "abstract": x["ref"]}
#         cands = [(x[n]["text"], x[n]["rouge1"]) for n in systems]
#         data["candidates"] = cands
#         with open(f"./CNNDM/two/val/{cnt}.json", "w") as out:
#             print(json.dumps(data), file=out)
#         cnt += 1
# print(cnt)

def collect_match_data():
    for i in range(4):
        match_dir = f"train_all_output.{i}.jsonl"
        with open(match_dir) as match_f:
            for x in match_f:
                yield x

def collect_bart_data():
    for i in range(5):
        bart_dir = f"/home/yixinl2/extractive-summarization/raw_data/bart/train.hypo.{i}.tokenized"
        with open(bart_dir) as bart_f:
            for x in bart_f:
                yield x

def collect_bart_train_data():
    for i in range(5):
        bart_dir = f"/home/yixinl2/extractive-summarization/raw_data/bart/train.{i}.source"
        with open(bart_dir) as bart_f:
            for x in bart_f:
                yield x

def zip_data():
    match_data = collect_match_data()
    bart_data = collect_bart_data()
    bart_source = collect_bart_train_data()
    cnt = 0
    with open("train.source", "w") as source, open("matchsum_train.jsonl", "w") as match:
        for (x, y, z) in zip(bart_source, bart_data, match_data):
            match_data = json.loads(z)
            if match_data["src"] is None:
                continue
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
            source.write(x)
            print(json.dumps(match_data["match"]), file=match)



def collect_data():
    cnt = 0
    bert_f = open("/home/yixinl2/extractive-summarization/raw_data/PreSumm/train_all.jsonl")
    match_data = collect_match_data()
    bart_data = collect_bart_data()
    for (x, y) in zip(bart_data, match_data):
        match_data = json.loads(y)
        bert_line = bert_f.readline()
        if match_data["src"] is None:
            continue
        yield (x, y, bert_line, cnt)
        cnt += 1
    bert_f.close()

def make_item(input):
    bart_line, match_line, bert_line, idx = input
    bart_line = bart_line.lower().strip().split(".")
    bart_result = []
    for s in bart_line:
        if len(s.strip()) > 0: 
            bart_result.append(s.strip() + " .")
    match_data = json.loads(match_line)
    if match_data["src"] is None:
        return
    output = dict()
    output["article"] = match_data["src"]
    bert_data = json.loads(bert_line)
    # ref = [" ".join(x) for x in bert_data["tgt"]]
    tgt_txt = [" ".join(x) for x in bert_data["tgt"]]
    ref = []
    for x in tgt_txt:
        if not x.endswith("."):
            x = x + " ."
        ref.append(x)
    output["abstract"] = ref
    # output["abstract"] = match_data["ref"]
    # scorer = RougeScorer(['rouge1'], use_stemmer=True)

    def compute_rouge(hyp):
        score = all_scorer.score("\n".join(output["abstract"]), "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3

    match_score = compute_rouge(match_data["match"])
    bart_score = compute_rouge(bart_result)
    output["candidates"] = [(match_data["match"], match_score), (bart_result, bart_score)]
    with open(f"./CNNDM/two_new/train/{idx}.json", "w") as out:
        print(json.dumps(output), file=out)

def make_data():
    unfinish_files = collect_data()
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(make_item, unfinish_files, chunksize=64))
    print("finish")

def make_file_list(split, threshold):
    fpath = f"./CNNDM/two/{split}"
    files = os.listdir(fpath)
    cnt = 0
    with open(f"./CNNDM/two/{split}-{threshold}.txt", "w") as out:
        for x in files:
            if random.random() < threshold:
                cnt += 1
                print(os.path.abspath(os.path.join(fpath, x)), file=out)
    print(cnt)

def make_small_file_list(split, ratio):
    fpath = f"./xsum/beam_pegasus/{split}"
    files = os.listdir(fpath)
    cnt = 0
    with open(f"./xsum/beam_pegasus/{split}-{ratio}-first.txt", "w") as out_1, open(f"./xsum/beam_pegasus/{split}-{ratio}-second.txt", "w") as out_2:
        for x in files:
            if random.random() < ratio:
                cnt += 1
                print(os.path.abspath(os.path.join(fpath, x)), file=out_1)
            else:
                print(os.path.abspath(os.path.join(fpath, x)), file=out_2)
    # with open(f"./CNNDM/beam/{split}-{ratio}.txt", "w") as out_1:
    #     for x in files:
    #         if random.random() < ratio:
    #             cnt += 1
    #             print(os.path.abspath(os.path.join(fpath, x)), file=out_1)
    print(cnt)

def split_file_list():
    fpath = f"./CNNDM/two/train_all"
    files = os.listdir(fpath)
    cnt = 0
    with open("./CNNDM/two/train_train.txt", "w") as train, open("./CNNDM/two/train_dev.txt", "w") as dev:
        for x in files:
            x = os.path.abspath(os.path.join(fpath, x))
            if random.randint(0, 20) == 0:
                cnt += 1
                print(x, file=dev)
            else:
                print(x, file=train)
    print(cnt)

def check():
    cnt = 0
    bert_f = open("/home/yixinl2/extractive-summarization/raw_data/PreSumm/train_all.jsonl")
    match_score = 0
    for i in range(4):
        # bart_dir = f"/home/yixinl2/extractive-summarization/raw_data/bart/train.hypo.{i}.tokenized"
        match_dir = f"train_all_output.{i}.jsonl"
        with open(match_dir) as match_f:
            for y in match_f:
                if cnt % 100 == 0:
                    print(cnt)
                match_data = json.loads(y)
                bert_line = bert_f.readline()
                if match_data["src"] is None:
                    continue
                bert_data = json.loads(bert_line)
                # ref = [" ".join(x) for x in bert_data["tgt"]]
                output = dict()
                tgt_txt = [" ".join(x) for x in bert_data["tgt"]]
                ref = []
                for x in tgt_txt:
                    if not x.endswith("."):
                        x = x + " ."
                    ref.append(x)
                output["abstract"] = ref
                # output["abstract"] = match_data["ref"]
                scorer = RougeScorer(['rouge1'], use_stemmer=True)

                def compute_rouge(hyp):
                    score = scorer.score("\n".join(output["abstract"]), "\n".join(hyp))
                    return score["rouge1"].fmeasure

                match_score += compute_rouge(match_data["match"])
                cnt += 1
                if cnt == 10000:
                    print(output["abstract"])
                    print(match_data)
                    # quit()
                if cnt > 10000:
                    break
    print(match_score / cnt)
    bert_f.close()

def combine_guided():
    source_path = "./CNNDM/two_mat/val"
    tgt_path = "./CNNDM/three_mat/val"
    nums = len(os.listdir(source_path))
    # scorer = RougeScorer(['rouge1'], use_stemmer=True)
    with open("val.ziyi.tokenized") as f_ziyi:
        for i in range(nums):
            if i % 1000 == 0:
                print(i)
            with open(os.path.join(source_path, f"{i}.json")) as f:
                data = json.load(f)
            bart_line = f_ziyi.readline()
            bart_line = bart_line.lower().strip().split(".")
            bart_result = []
            for s in bart_line:
                if len(s.strip()) > 0: 
                    bart_result.append(s.strip() + " .")
            # score = scorer.score("\n".join(data["abstract"]), "\n".join(bart_result))["rouge1"].fmeasure
            score = all_scorer.score("\n".join(data["abstract"]), "\n".join(bart_result))
            score = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
            # score = scorer.score("\n".join(data["abstract"]), "\n".join(bart_result))["rouge1"].fmeasure
            data["candidates"].append((bart_result, score))
            with open(os.path.join(tgt_path, f"{i}.json"), "w") as f:
                json.dump(data, f)

def combine_guided_test():
    source_path = "./CNNDM/two_mat/test"
    tgt_path = "./CNNDM/three_mat/test"
    nums = len(os.listdir(source_path))
    # scorer = RougeScorer(['rouge1'], use_stemmer=True)
    with open("topthree.jsonl") as f_ziyi:
        for i in range(nums):
            if i % 1000 == 0:
                print(i)
            with open(os.path.join(source_path, f"{i}.json")) as f:
                data = json.load(f)
            bart = json.loads(f_ziyi.readline())
            bart_result = bart["guide"]
            score = all_scorer.score("\n".join(data["abstract"]), "\n".join(bart_result))
            score = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
            data["candidates"].append((bart_result, score))
            with open(os.path.join(tgt_path, f"{i}.json"), "w") as f:
                json.dump(data, f)

def make_trigrams(x):
    x = x.split(" ")
    trigrams = set()
    for i in range(len(x) - 2):
        trigrams.add(" ".join(x[i:i+3]))
    return trigrams

def exist_trigram(s, c):
    for x in s:
        a = make_trigrams(x)
        b = make_trigrams(c)
        if not a.isdisjoint(b):
            return True
    return False

def make_rank(fname):
    src_path = "./CNNDM/two_mat/train"
    tgt_path = "./CNNDM/rank_mat/train"
    with open(os.path.join(src_path, fname)) as f:
        data = json.load(f)
    match = data["candidates"][0][0]
    bart = data["candidates"][1][0]
    match, bart = match[:4], bart[:4]
    sents = match + bart
    sent_id = [_ for _ in range(len(sents))]
    indices = list(combinations(sent_id, 3))
    # indices += list(combinations(sent_id, 3))
    if len(sent_id) < 2:
        indices = [sent_id]
    cands = []
    scores = []
    ref = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(ref, "\n".join(hyp))
        score = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
        return score

    for ids in indices:
        cand = []
        flag = False
        ids = list(ids)
        ids.sort()
        for id in ids:
            if exist_trigram(cand, sents[id]):
                flag = True
                break
            cand.append(sents[id])
        if not flag:
            scores.append(compute_rouge(cand))
            cands.append(ids)
    if len(cands) == 0:
        cands.append([_ for _ in range(len(match))])
        cands.append([_ for _ in range(len(match), len(bart))])
        scores.append(compute_rouge(match))
        scores.append(compute_rouge(bart))
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": cands,
        "scores": scores,
        "cand_sents": sents
        }
    with open(os.path.join(tgt_path, fname), "w") as f:
        json.dump(output, f)
    
def make_rank_data():
    unfinish_files = os.listdir("./CNNDM/two_mat/train")
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(make_rank, unfinish_files, chunksize=64))
    print("finish")

def split(x, lower=True):
    if lower:
        x = x.lower()
    x = x.strip().split(".")
    result = []
    for s in x:
        if len(s.strip()) > 0: 
            result.append(s.strip() + " .")
    return result

def collect_cand_data():
    result = []
    # for i in range(5):
    #     bart_dir = f"/home/yixinl2/extractive-summarization/raw_data/bart/train.cand.{i}.tokenized"
    #     with open(bart_dir) as bart_f:
    #         for x in bart_f:
    #             result.append(split(x))
    #             if len(result) == 4:
    #                 yield result
    #                 result = []
    # bart_dir = f"/home/yixinl2/extractive-summarization/raw_data/bart/test.cand.tokenized"
    bart_dir = "./val.beam.ziyi.tokenized"
    with open(bart_dir) as bart_f:
        for x in bart_f:
            result.append(split(x))
            if len(result) == 4:
                yield result
                result = []

def collect_beam_data():
    bert_f = open("/home/yixinl2/extractive-summarization/raw_data/PreSumm/train_all.jsonl")
    match_data = collect_match_data()
    beam_data = collect_cand_data()
    cnt = 0
    for (x, y) in zip(beam_data, match_data):
        match_item = json.loads(y)
        bert_line = bert_f.readline()
        if match_item["src"] is None:
            continue
        yield (x, y, bert_line, cnt)
        cnt += 1
    bert_f.close()

# def make_beam(input):
#     bart_line, match_line, bert_line, idx = input
#     match_data = json.loads(match_line)
#     if match_data["src"] is None:
#         return
#     output = dict()
#     output["article"] = match_data["src"]
#     bert_data = json.loads(bert_line)
#     # ref = [" ".join(x) for x in bert_data["tgt"]]
#     tgt_txt = [" ".join(x) for x in bert_data["tgt"]]
#     ref = []
#     for x in tgt_txt:
#         if not x.endswith("."):
#             x = x + " ."
#         ref.append(x)
#     output["abstract"] = ref
#     # output["abstract"] = match_data["ref"]
#     scorer = RougeScorer(['rouge1'], use_stemmer=True)

#     def compute_rouge(hyp):
#         score = scorer.score("\n".join(output["abstract"]), "\n".join(hyp))
#         return score["rouge1"].fmeasure

#     output["candidates"] = [(x, compute_rouge(x)) for x in bart_line]
#     with open(f"./CNNDM/beam/train/{idx}.json", "w") as out:
#         print(json.dumps(output), file=out)

def make_beam(input):
    src_path = "./CNNDM/two/val"
    idx, bart_line = input
    with open(os.path.join(src_path, f"{idx}.json")) as f:
        output = json.load(f)
    def compute_rouge(hyp):
        score = scorer.score("\n".join(output["abstract"]), "\n".join(hyp))
        return score["rouge1"].fmeasure

    output["candidates"] = [(x, compute_rouge(x)) for x in bart_line]
    with open(f"./CNNDM/beam_guide/val/{idx}.json", "w") as out:
        print(json.dumps(output), file=out)

def make_beam_data():
    # data = collect_beam_data()
    data = collect_cand_data()
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(make_beam, enumerate(data), chunksize=64))
    print("finish")

def collect_iter_data(split):
    source_dir = f"./result/main-two-0.1/{split}_output.jsonl"
    with open(source_dir) as f:
        for (i, x) in enumerate(f):
            yield (i, x)
            
def make_iter(input, split):
    # match, guide
    guided_dir = f"./CNNDM/three/{split}"
    tgt_dir = f"./CNNDM/iter/{split}"
    i, decoded = input
    decoded = json.loads(decoded)
    with open(os.path.join(guided_dir, f"{i}.json")) as f:
        data = json.load(f)
    guided_item = data["candidates"][2]
    candidates = [decoded, guided_item[0]]
    def compute_rouge(hyp):
        score = scorer.score("\n".join(data["abstract"]), "\n".join(hyp))
        return score["rouge1"].fmeasure
    data["candidates"] = [(x, compute_rouge(x)) for x in candidates]
    with open(os.path.join(tgt_dir, f"{i}.json"), "w") as f:
        json.dump(data, f)

def make_iter_data(split):
    # data = collect_beam_data()
    data = collect_iter_data(split)
    f_n = partial(make_iter, split=split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(f_n, data, chunksize=64))
    print("finish")

def split_data(split):
    files = os.listdir(f"./CNNDM/three/{split}")
    for i in range(len(files)):
        with open(os.path.join(f"./CNNDM/three/{split}", f"{i}.json")) as f:
            data = json.load(f)
            data["candidates"] = [data["candidates"][0], data["candidates"][2]] # matchsum and guide
            if i % 10000 == 0:
                print(i)
            with open(os.path.join(f"./CNNDM/mix/{split}", f"{i}.json"), "w") as out:
                json.dump(data, out)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

def build_label(fname):
    fdir = "./CNNDM/three/train"
    with open(os.path.join(fdir, fname)) as f:
        data = json.load(f)
    data["label"] = greedy_selection(data["article"], data["abstract"], 3)
    with open(os.path.join(fdir, fname), "w") as f:
        json.dump(data, f)

def make_label():
    # data = collect_beam_data()
    fdir = "./CNNDM/three/train"
    files = os.listdir(fdir)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_label, files, chunksize=64))
    print("finish")

def collect_pre_data():
    fdir = "./CNNDM/three/test"
    num = len(os.listdir(fdir))
    files = [os.path.join(fdir, f"{i}.json") for i in range(num)]
    # with open("./CNNDM/three/train-0.5-first.txt") as f:
    #     files = f.readlines()
    with open("../extractive-summarization/test_idx.jsonl") as f:
        for (i, x) in enumerate(f):
            yield (files[i].strip(), x)

def build_match(input):
    fname, indices = input
    fdir = "./CNNDM/base/test"
    with open(fname) as f:
        data = json.load(f)
    sent_id = json.loads(indices)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    sents = data["article"]
    indices = list(combinations(sent_id, 3)) + list(combinations(sent_id, 2))
    if len(sent_id) < 2:
        indices = [sent_id]
    cands = []
    scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [sents[id] for id in ids]
        scores.append(compute_rouge(cand))
        cands.append(ids)
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": cands,
        "scores": scores,
        "cand_sents": sents
        }
    fname = fname.split("/")[-1]
    with open(os.path.join(fdir, fname), "w") as f:
        json.dump(output, f)

def make_match_data():
    data = collect_pre_data()
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_match, data, chunksize=64))
    print("finish")

def transfer_list():
    src_dir = "./CNNDM/three/train-0.5-first.txt"
    tgt_dir = "./CNNDM/base/train-0.5-first.txt"
    with open(src_dir) as f, open(tgt_dir, "w") as out:
        for x in f:
            x = x.strip()
            x = x.split("/")[-1]
            print(os.path.abspath(os.path.join("./CNNDM/base/train", x)), file=out)

def collect_second_data():
    # fdir = "./CNNDM/three/val
    with open("./CNNDM/three/train-0.5-first.txt") as f:
        files = f.readlines()
    # fdir = "./CNNDM/three/val"
    # num = len(os.listdir(fdir))
    # files = [os.path.join(fdir, f"{i}.json") for i in range(num)]
    with open("./result/base/train-first.jsonl") as f:
        for (i, x) in enumerate(f):
            yield (files[i].strip(), x)

def build_second(input):
    fname, output = input
    fdir = "./CNNDM/base-second/train"
    with open(fname) as f:
        data = json.load(f)
    ref = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = scorer.score(ref, "\n".join(hyp))
        return score["rouge1"].fmeasure
    match = json.loads(output)
    data["candidates"] = [(match, compute_rouge(match)), data["candidates"][2]]
    fname = fname.split("/")[-1]
    with open(os.path.join(fdir, fname), "w") as f:
        json.dump(data, f)

def make_second_data():
    data = collect_second_data()
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_second, data, chunksize=64))
    print("finish")

def transfer_data():
    src_dir = "./CNNDM/three/train"
    base_dir = "./CNNDM/base-second/train"
    tgt_dir = "./CNNDM/three/train-unbiased"
    files = os.listdir(src_dir)
    for (i, x) in enumerate(files):
        if i % 10000 == 0:
            print(i)
        with open(os.path.join(src_dir, x)) as f:
            data = json.load(f)
        with open(os.path.join(base_dir, x)) as f:
            base_data = json.load(f)
        data["candidates"] = [base_data["candidates"][0], data["candidates"][1], base_data["candidates"][1]]
        with open(os.path.join(tgt_dir, x), "w") as f:
            json.dump(data, f)

def collect_origin_data(split):
    bert_dir = f"/home/yixinl2/extractive-summarization/raw_data/PreSumm/{split}_result.jsonl"
    cnt = 0
    with open(bert_dir) as f:
        for x in f:
            data = json.loads(x)
            if len(data["src_txt"]) != 0:
                yield (cnt, x)
                cnt += 1

def build_origin(input):
    idx, line = input
    src_dir = "./CNNDM/three/test"
    tgt_dir = "./CNNDM/origin/test"
    with open(os.path.join(src_dir, f"{idx}.json")) as f:
        data = json.load(f)
    bert = json.loads(line)
    sent_id = []
    sent_num = len(bert["src_txt"])
    for id in bert["selected_ids"]:
        if id < sent_num:
            sent_id.append(id)
        if len(sent_id) == 5:
            break
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    sents = data["article"]
    indices = list(combinations(sent_id, 3)) + list(combinations(sent_id, 2))
    if len(sent_id) < 2:
        indices = [sent_id]
    cands = []
    scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [sents[id] for id in ids]
        scores.append(compute_rouge(cand))
        cands.append(ids)
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": cands,
        "scores": scores,
        "cand_sents": sents
        }
    with open(os.path.join(tgt_dir, f"{idx}.json"), "w") as f:
        json.dump(output, f)


def make_origin_data(split):
    data = collect_origin_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_origin, data, chunksize=64))
    print("finish")

def other_data(fdir, tdir):
    with open(fdir) as f:
        for (i, x) in enumerate(f):
            if i % 1000 == 0:
                print(i)
            data = json.loads(x)
            output = {
                "article": data["text"], 
                "abstract": data["summary"],
                "indices": data["indices"],
                "scores": data["score"],
                "cand_sents": data["text"],
                "label": sorted(data["ext_idx"])
            }
            with open(os.path.join(tdir, f"{i}.json"), "w") as out:
                json.dump(output, out)

def collect_xsum_data(split):
    base_dir = f"./xsum/base/{split}"
    tgt_dir = f"./xsum/two/{split}"
    with open(f"./data/xsum/xsum.{split}.out.tokenized") as f:
        bart = [x.strip().lower() for x in f]
    with open(f"./data/xsum/{split}.match.idx") as f:
        idx = json.load(f)
    with open(f"./data/xsum.{split}.match.jsonl") as f:
        for i, match in enumerate(f):
            yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), bart[idx[i]], match)

def build_xsum(input):
    src_dir, tgt_dir, bart, match = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    bart = sent_detector.tokenize(bart)
    match = json.loads(match)
    candidates = [(match, compute_rouge(match)), (bart, compute_rouge(bart))]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_data(split):
    data = collect_xsum_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_xsum, data, chunksize=64))
    print("finish")

def collect_other_data(split):
    base_dir = f"./pubmed/base/{split}"
    tgt_dir = f"./pubmed/two/{split}"
    with open(f"./data/pubmed/pubmed.{split}.out.tokenized") as f:
        bart = [x.strip().lower() for x in f]
    with open(f"./data/pubmed.{split}.match.jsonl") as f:
        for i, match in enumerate(f):
            yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), bart[i], match)

def build_other(input):
    src_dir, tgt_dir, bart, match = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    bart = sent_detector.tokenize(bart)
    match = json.loads(match)
    candidates = [(match, compute_rouge(match)), (bart, compute_rouge(bart))]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_other_data(split):
    data = collect_other_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_other, data, chunksize=64))
    print("finish")

def gather_beam_data(fdir):
    sents = []
    with open(fdir) as f:
        for x in f:
            x = x.strip().lower()
            sents.append(x)
            if len(sents) == 4:
                yield sents
                sents = []


def collect_xsum_beam_data(split):
    base_dir = f"./data/{split}_wikihow.jsonl"
    tgt_dir = f"./wikihow/beam/{split}"
    beam_data = gather_beam_data(f"./data/wikihow/wikihow.{split}.beam.out.tokenized")
    # with open(f"./data/pubmed/wikihow.{split}.beam.out.tokenized") as f:
    #     _bart = [x.strip().lower() for x in f]
    # with open(base_dir) as f:
    #     lines = f.readlines()
    # with open(f"./data/pubmed/pubmed.{split}.beam.new.out.tokenized") as f:
    #     for x in f:
    #         _bart.append(x.strip().lower())
    # bart = []
    # sents = []
    # nums = len(os.listdir(base_dir))
    # for x in _bart:
    #     sents.append(x)
    #     if len(sents) == 4:
    #         bart.append(sents)
    #         sents = []
    # print(nums)
    # print(len(bart))
    # with open(f"./data/xsum/{split}.match.idx") as f:
    #     idx = json.load(f)
    with open(base_dir) as f:
        for (i, (beam, line)) in enumerate(zip(beam_data, f)):
            yield (line, os.path.join(tgt_dir, f"{i}.json"), beam)

def build_beam_xsum(input):
    src_dir, tgt_dir, bart = input
    data = json.loads(src_dir)
    abstract = "\n".join(data["summary"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    bart = [sent_detector.tokenize(x) for x in bart]
    candidates = [(x, compute_rouge(x)) for x in bart]
    output = {
        "article": data["text"], 
        "abstract": data["summary"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_beam_data(split):
    data = collect_xsum_beam_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_beam_xsum, data, chunksize=64))
    print("finish")

def collect_xsum_new_data(split):
    base_dir = f"./xsum/base/{split}"
    tgt_dir = f"./xsum/new/{split}"
    bart = []
    pegasus = []
    for i in range(1, 4):
        with open(f"./data/xsum/{split}.bart.out.tokenized") as fb, open(f"./data/xsum/{split}.pegasus.out.tokenized") as fp: 
            for (x, y) in zip(fb, fp):
                bart.append(x.strip().lower())
                pegasus.append(y.strip().lower())
    nums = len(os.listdir(base_dir))
    with open(f"./data/xsum/{split}.match.idx") as f:
        idx = json.load(f)
    for i in range(nums):
        yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), bart[idx[i]], pegasus[idx[i]])

def build_new_xsum(input):
    src_dir, tgt_dir, bart, pegasus = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    bart = [bart]
    pegasus = [pegasus]
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    candidates = [(bart, compute_rouge(bart)), (pegasus, compute_rouge(pegasus))] # bart, pegasus
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_new_data(split):
    data = collect_xsum_new_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_new_xsum, data, chunksize=64))
    print("finish")

def collect_oracle_data(split):
    src_dir = f"./pubmed/beam/{split}"
    tgt_dir = f"./pubmed/oracle/{split}"
    num = len(os.listdir(src_dir))
    for i in range(num):
        yield (os.path.join(src_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"))

def build_oracle(input):
    src_dir, tgt_dir = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    sent_scores = [compute_rouge([x]) for x in data["article"]]
    max_ids = np.argsort(-np.array(sent_scores)).tolist()
    sent_id = max_ids[:7]
    sents = data["article"]
    # indices = list(combinations(sent_id, 3)) + list(combinations(sent_id, 4)) + list(combinations(sent_id, 5))
    indices = list(combinations(sent_id, 6))
    if len(sent_id) < 6:
        indices = [sent_id]
    cands = []
    scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [sents[id] for id in ids]
        scores.append(compute_rouge(cand))
        cands.append(ids)
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": cands,
        "scores": scores,
        "cand_sents": sents
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)


def make_oracle_data(split):
    data = collect_oracle_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_oracle, data, chunksize=64))
    print("finish")

def select_samples(x, max_num=30):
    if len(x) == 1:
        return [0]
    if len(x) == 2:
        return [0, 1]
    num = len(x)
    ids = np.random.choice(num - 2, size=min(num - 2, max_num - 2), replace=False) + 1
    ids = ids.tolist()
    ids.sort()
    ids.append(-1)
    ids = [0] + ids
    return ids 

def select_oracle_data(split):
    src_dir = f"./CNNDM/oracle/{split}"
    tgt_dir = f"./CNNDM/oracle_30/{split}"
    num = len(os.listdir(src_dir))
    for i in range(num):
        if i % 1000 == 0:
            print(i)
        with open(os.path.join(src_dir, f"{i}.json")) as f:
            data = json.load(f)
        ids = select_samples(data["scores"])
        data["scores"] = [data["scores"][i] for i in ids]
        data["indices"] = [data["indices"][i] for i in ids]
        with open(os.path.join(tgt_dir, f"{i}.json"), "w") as f:
            json.dump(data, f)
        
def collect_combine_data(split):
    bart_dir = f"./xsum/beam_bart/{split}"
    pegasus_dir = f"./xsum/beam_pegasus/{split}"
    tgt_dir = f"./xsum/combine/{split}"
    num = len(os.listdir(bart_dir))
    for i in range(num):
        yield (os.path.join(bart_dir, f"{i}.json"), os.path.join(pegasus_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"))

def build_combine(input):
    bart_dir, pegasus_dir, tgt_dir = input
    with open(bart_dir) as f:
        data = json.load(f)
    with open(pegasus_dir) as f:
        pegasus_data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    sent_scores = [compute_rouge([x]) for x in data["article"]]
    max_ids = np.argsort(-np.array(sent_scores)).tolist()
    sent_id = max_ids[:5]
    sents = data["article"]
    indices = list(combinations(sent_id, 1)) + list(combinations(sent_id, 2))
    if len(sent_id) < 2:
        indices = [sent_id]
    scores = [x[1] for x in data["candidates"]] + [x[1] for x in pegasus_data["candidates"]]
    candidates = [x[0] for x in data["candidates"]] + [x[0] for x in pegasus_data["candidates"]]
    cands = []
    _scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [sents[id] for id in ids]
        _scores.append(compute_rouge(cand))
        cands.append(cand)
    tmp = zip(cands, _scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    tmp = tmp[:6]
    cands = [y[0] for y in tmp]
    _scores = [y[1] for y in tmp]
    candidates.extend(cands)
    scores.extend(_scores)
    candidates = [(candidates[i], scores[i]) for i in range(len(scores))]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_combine_data(split):
    data = collect_combine_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_combine, data, chunksize=64))
    print("finish")

def make_mixture_file_list(ratio=0.5, train_ratio=0.2):
    dataset = "./CNNDM/three"
    fpath = os.path.join(dataset, "val")
    files = os.listdir(fpath)
    cnt = 0
    with open(os.path.join(dataset, "mixture-train.txt"), "w") as out_1, open(os.path.join(dataset, "mixture-val.txt"), "w") as out_2:
        for x in files:
            if random.random() < ratio:
                cnt += 1
                print(os.path.abspath(os.path.join(fpath, x)), file=out_1)
            else:
                print(os.path.abspath(os.path.join(fpath, x)), file=out_2)
        fpath = os.path.join(dataset, "train")
        files = os.listdir(fpath)
        for x in files:
            if random.random() < train_ratio:
                cnt += 1
                print(os.path.abspath(os.path.join(fpath, x)), file=out_1)
    # with open(f"./CNNDM/beam/{split}-{ratio}.txt", "w") as out_1:
    #     for x in files:
    #         if random.random() < ratio:
    #             cnt += 1
    #             print(os.path.abspath(os.path.join(fpath, x)), file=out_1)
    print(cnt)

def collect_xsum_aug_data(split):
    base_dir = f"./xsum/new/{split}"
    tgt_dir = f"./xsum/aug_pegasus/{split}"
    fdir = "./eda_nlp/code/output.txt"
    lines = []
    i = 0
    with open(fdir) as f:
        for x in f:
            lines.append([x.strip()])
            if len(lines) == 4:
                yield (os.path.join(base_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), lines)
                lines = []
                i += 1

def build_aug_xsum(input):
    src_dir, tgt_dir, cands = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
    candidates = [(x, compute_rouge(x)) for x in cands]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_xsum_aug_data(split):
    data = collect_xsum_aug_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_aug_xsum, data, chunksize=64))
    print("finish")

def collect_cnndm_pre_data(split):
    tgt_dir = f"./CNNDM/pre/{split}"
    fdir = f"../MatchSum/data/{split}_CNNDM_bert.jsonl"
    with open(fdir) as f:
        for (i, x) in enumerate(f):
            yield (x, os.path.join(tgt_dir, f"{i}.json"))

def build_pre_cnndm(input):
    src_dir, tgt_dir = input
    data = json.loads(src_dir)
    output = {
        "article": data["text"], 
        "abstract": data["summary"],
        "indices": data["indices"],
        "scores": data["score"],
        "cand_sents": data["text"]
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_cnndm_pre_data(split):
    data = collect_cnndm_pre_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_pre_cnndm, data, chunksize=64))
    print("finish")

def collect_diverse_data(split):
    src_dir = f"./wikihow/beam/{split}"
    tgt_dir = f"./wikihow/diverse/{split}"
    cands = []
    sents = []
    with open(f"./data/wikihow/{split}.diverse.out.tokenized") as f:
        for x in f:
            x = x.strip().lower()
            sents.append(x)
            if len(sents) == 16:
                cands.append(sents)
                sents = []
    nums = len(cands)
    print(nums)
    for i in range(nums):
        yield (os.path.join(src_dir, f"{i}.json"), os.path.join(tgt_dir, f"{i}.json"), cands[i])

def build_diverse(input):
    src_dir, tgt_dir, cands = input
    cands = list(set(cands))
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    cands = [(x, compute_rouge([x])) for x in cands]
    cands = sorted(cands, key=lambda x: x[1], reverse=True)
    indices = [[i] for i in range(len(cands))]
    scores = [x[1] for x in cands]
    cand_sents = [x[0] for x in cands]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "indices": indices,
        "scores": scores,
        "cand_sents": cand_sents
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_diverse_data(split):
    data = collect_diverse_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_diverse, data, chunksize=64))
    print("finish")

def collect_pretrain_data(split):
    src_dir = f"./CNNDM/pre/{split}"
    tgt_dir = f"./CNNDM/pretrain/{split}"
    # num = len(os.listdir(src_dir))
    files = os.listdir(src_dir)
    cnt = 0
    for (i, x) in enumerate(files):
        with open(os.path.join(src_dir, f"{i}.json")) as f:
            data = json.load(f)
        if len(data["article"]) < 7:
            continue
        yield (os.path.join(src_dir, f"{i}.json"), os.path.join(tgt_dir, f"{cnt}.json"))
        cnt += 1
    print(cnt)

def build_pretrain(input):
    src_dir, tgt_dir = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["article"][:3])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    cand_sents = data["article"][3:]
    sent_scores = [compute_rouge([x]) for x in cand_sents]
    max_ids = np.argsort(-np.array(sent_scores)).tolist()
    sent_id = max_ids[:5]
    indices = list(combinations(sent_id, 2)) + list(combinations(sent_id, 3))
    # indices = list(combinations(sent_id, 6))
    if len(sent_id) < 3:
        indices = [sent_id]
    cands = []
    scores = []
    for ids in indices:
        ids = list(ids)
        ids.sort()
        cand = [cand_sents[id] for id in ids]
        scores.append(compute_rouge(cand))
        cands.append(ids)
    tmp = zip(cands, scores)
    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)
    cands = [y[0] for y in tmp]
    scores = [y[1] for y in tmp]
    output = {
        "article": cand_sents, 
        "abstract": data["article"][:3],
        "indices": cands,
        "scores": scores,
        "cand_sents": cand_sents,
        "real_abs": data["abstract"]
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)


def make_pretrain_data(split):
    data = collect_pretrain_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_pretrain, data, chunksize=64))
    print("finish")

def collect_new_data(split):
    base_dir = f"./CNNDM/three/{split}"
    tgt_dir = f"./CNNDM/two_mat/{split}"
    cnt = 0
    for j in range(3):
        fdir = f"./result/cnndm_mat/{split}.{j}.jsonl"
        with open(fdir) as f:
            for (i, x) in enumerate(f):
                match_data = json.loads(x)
                if match_data["src"] is None:
                    continue
                yield (os.path.join(base_dir, f"{cnt}.json"), os.path.join(tgt_dir, f"{cnt}.json"), x)
                cnt += 1

def build_new(input):
    src_dir, tgt_dir, line = input
    match_data = json.loads(line)
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    match_data = match_data["match"]
    cands = [match_data, data["candidates"][1][0]]
    candidates = [(x, compute_rouge(x)) for x in cands]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_new_data(split):
    data = collect_new_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_new, data, chunksize=64))
    print("finish")

def collect_guide_data(split):
    base_dir = f"./CNNDM/two_mat/{split}"
    tgt_dir = f"./CNNDM/three_mat/{split}"
    cnt = 0
    fdir = f"{split}.ziyi.tokenized"
    with open(fdir) as f:
        for (i, x) in enumerate(f):
            yield (os.path.join(base_dir, f"{cnt}.json"), os.path.join(tgt_dir, f"{cnt}.json"), x)
            cnt += 1

def build_guide(input):
    src_dir, tgt_dir, bart_line = input
    with open(src_dir) as f:
        data = json.load(f)
    bart_line = bart_line.lower().strip().split(".")
    bart_result = []
    for s in bart_line:
        if len(s.strip()) > 0: 
            bart_result.append(s.strip() + " .")
    score = all_scorer.score("\n".join(data["abstract"]), "\n".join(bart_result))
    score = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
    data["candidates"].append((bart_result, score))
    with open(tgt_dir, "w") as f:
        json.dump(data, f)

def make_guide_data(split):
    data = collect_guide_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_guide, data, chunksize=64))
    print("finish")

def get_orig_data():
    with open("/home/yixinl2/extractive-summarization/raw_data/PreSumm/test_result.jsonl") as f:
        lines = f.readlines()
    for (i, x) in enumerate(lines):
        data = json.loads(x)
        sent_id = []
        cnt = 0
        sent_num = len(data["src_txt"])
        for id in data["selected_ids"]:
            if id < sent_num:
                sent_id.append(id)
            if len(sent_id) == 5:
                break
        indices = list(combinations(sent_id, 2))
        indices += list(combinations(sent_id, 3))
        output = dict()
        output["article"] = data["src_txt"]
        output["cand_sents"] = data["src_txt"]
        output["indices"] = indices
        ref = data["tgt_txt"].split("<q>")
        ref = [r.strip() + " ." for r in ref]
        output["abstract"] = ref
        with open(f"./CNNDM/pre/test_align/{i}.json", "w") as f:
            json.dump(output, f)

def collect_l2_data(split):
    base_dir = f"./xsum/beam_pegasus/{split}"
    tgt_dir = f"./xsum/beam_l2/{split}"
    files = os.listdir(base_dir)
    for x in files:
        yield (os.path.join(base_dir, x), os.path.join(tgt_dir, x))

def build_l2(input):
    src_dir, tgt_dir = input
    with open(src_dir) as f:
        data = json.load(f)
    abstract = "\n".join(data["abstract"])
    def compute_rouge(hyp):
        score = all_scorer.score(abstract, "\n".join(hyp))
        return 2 * (score["rouge1"].fmeasure * score["rouge2"].fmeasure) / (score["rouge1"].fmeasure + score["rouge2"].fmeasure + 1e-10)
    candidates = [(x[0], compute_rouge(x[0])) for x in data["candidates"][:4]]
    output = {
        "article": data["article"], 
        "abstract": data["abstract"],
        "candidates": candidates
        }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)

def make_l2_data(split):
    data = collect_l2_data(split)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_l2, data, chunksize=64))
    print("finish")

if __name__ == "__main__":
    # make_data()
    # zip_data()
    # make_file_list("train_all", 0.05)
    # split_file_list()
    # check()
    # with open("/home/yixinl2/extractive-summarization/raw_data/PreSumm/train_all_result.jsonl") as f:
    #     for (i, x) in enumerate(f):
    #         if i == 9999:
    #             print(x)
    # combine_guided_test()
    # make_rank_data()
    # combine_guided()
    # make_beam_data()
    # make_iter_data("test")
    # make_file_list("train_all", 0.8)
    # make_label()
    # split_data("val")
    # make_match_data()
    # transfer_list()
    # make_second_data()
    # # transfer_data()
    # make_origin_data("test")
    # other_data("./data/test_pubmed.jsonl", "./pubmed/base/test")
    # make_xsum_data("train")
    # make_other_data("train")
    # make_other_data("val")
    # make_xsum_beam_data("train")
    # make_xsum_beam_data("val")
    # make_xsum_beam_data("train")
    # make_xsum_beam_data("val")
    # make_xsum_new_data("val")
    # make_xsum_new_data("val")
    # make_oracle_data("test")
    # make_oracle_data("train")
    # make_oracle_data("val")
    # select_oracle_data("test")
    # select_oracle_data("val")
    # select_oracle_data("train")
    # make_combine_data("test")
    # make_combine_data("train")
    # make_combine_data("val")
    # make_mixture_file_list()
    # make_xsum_aug_data("train")
    # make_cnndm_pre_data("train")
    # make_cnndm_pre_data("val")
    # make_oracle_data("test")
    # make_diverse_data("test")
    # make_diverse_data("train")
    # make_diverse_data("val")
    # make_oracle_data("val")
    # make_pretrain_data("train")
    # make_pretrain_data("val")
    # make_new_data("train")
    # make_new_data("val")
    # make_new_data("test")
    # get_orig_data()
    # make_guide_data("train")
    # make_guide_data("test")
    make_l2_data("train")
    make_l2_data("val")



    

