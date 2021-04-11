import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pickle
import time
import numpy as np
import os
import json
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BertModel, BertTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, collate_mp_ids, RefactoringDataset, RefactoringIDsDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from model import RankingLoss, Refactor
import math
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.num_layers = getattr(args, 'num_layers', 2)  # transformer layers
    args.epoch = getattr(args, 'epoch', 5)  # number of epochs
    args.report_freq = getattr(args, "report_freq", 100)  # report frequency
    args.test_freq = getattr(args, "test_freq", 100)  # test frequency
    args.accumulate_step = getattr(args, "accumulate_step", 12)
    args.margin = getattr(args, "margin", 0.01)
    args.gold_margin = getattr(args, "gold_margin", 0)
    args.cand_weight = getattr(args, "cand_weight", 1)
    args.gold_weight = getattr(args, "gold_weight", 1)
    args.model_type = getattr(args, "model_type", 'bert-base-uncased')
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 970903)
    args.no_gold = getattr(args, "no_gold", False)
    args.pretrained = getattr(args, "pretrained", None)
    args.max_lr = getattr(args, "max_lr", 2e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "pre")
    args.dataset = getattr(args, "dataset", "CNNDM")
    args.use_ids = getattr(args, "use_ids", True)  # set true for pretraining
    args.max_len = getattr(args, "max_len", 512)
    args.max_num = getattr(args, "max_num", 4)  # max number of candidates


def evaluation(args):
    # load data
    base_setting(args)
    tok = BertTokenizer.from_pretrained(args.model_type)
    if args.use_ids:
        collate_fn = partial(collate_mp_ids, pad_token_id=tok.pad_token_id, is_test=True)
        test_set = RefactoringIDsDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, maxlen=512, is_sorted=False)
    else:
        collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
        test_set = RefactoringDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num)
    dataloader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = Refactor(model_path, num_layers=args.num_layers)
    if args.cuda:
        model = model.cuda()
    
    if args.model_pt is not None:
        state_dict = torch.load(args.model_pt, map_location=f'cuda:{args.gpuid[0]}')
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
    model.eval()
    
    model_name = args.model_name

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    mkdir("./result/%s"%model_name)
    mkdir("./result/%s/reference"%model_name)
    mkdir("./result/%s/candidate"%model_name)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    acc = 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            samples = batch["data"]
            output = model(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity.cpu().numpy()
            if i % 100 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            acc += (max_ids == batch["scores"].cpu().numpy().argmax(1)).sum()
            for j in range(similarity.shape[0]):
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                with open("./result/%s/candidate/%d.dec"%(model_name, cnt), "w") as f:
                    for s in sents:
                        print(s, file=f)
                with open("./result/%s/reference/%d.ref"%(model_name, cnt), "w") as f:
                    for s in sample["abstract"]:
                        print(s, file=f)
                cnt += 1
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    print(f"accuracy: {acc / cnt}")
    print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))


def test(dataloader, model, args, gpuid):
    model.eval()
    loss = 0
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            samples = batch["data"]
            output = model(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity.cpu().numpy()
            if i % 1000 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    model.train()
    loss = 1 - ((rouge1 + rouge2 + rougeLsum) / 3)
    print(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    
    if len(args.gpuid) > 1:
        loss = torch.FloatTensor([loss]).to(gpuid)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        loss = loss.item() / len(args.gpuid)
    return loss


def run(rank, args):
    base_setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        recorder = Recorder(args.log)
    tok = BertTokenizer.from_pretrained(args.model_type)
    if args.use_ids:
        collate_fn = partial(collate_mp_ids, pad_token_id=tok.pad_token_id, is_test=False)
        collate_fn_val = partial(collate_mp_ids, pad_token_id=tok.pad_token_id, is_test=True)
        train_set = RefactoringIDsDataset(f"./{args.dataset}/{args.datatype}/train", args.model_type, maxlen=args.max_len, max_num=args.max_num)
        val_set = RefactoringIDsDataset(f"./{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, maxlen=512, is_sorted=False)
    else:
        collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
        collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
        train_set = RefactoringDataset(f"./{args.dataset}/{args.datatype}/train", args.model_type, maxlen=args.max_len, maxnum=args.max_num)
        val_set = RefactoringDataset(f"./{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = Refactor(model_path, num_layers=args.num_layers)

    if args.model_pt is not None:
        model.load_state_dict(torch.load(args.model_pt, map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if len(args.gpuid) == 1:
            model = model.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=True)
    model.train()
    init_lr = args.max_lr / args.warmup_steps
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [model], __file__)
    minimum_loss = 100
    all_step_cnt = 0
    # start training
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        step_cnt = 0
        steps = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            output = model(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            loss = args.scale * RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight, no_gold=args.no_gold)
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:               
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                steps += 1
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
            if steps % args.report_freq == 0 and step_cnt == 0 and is_master:
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f"%(epoch+1, steps, 
                 avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0
            del similarity, gold_similarity, loss

            if all_step_cnt % args.test_freq == 0 and all_step_cnt != 0 and step_cnt == 0:
                loss = test(val_dataloader, model, args, gpuid)
                if loss < minimum_loss and is_master:
                    minimum_loss = loss
                    if is_mp:
                        recorder.save(model.module, "model.bin")
                    else:
                        recorder.save(model, "model.bin")
                    recorder.save(optimizer, "optimizer.bin")
                    recorder.print("best - epoch: %d, batch: %d"%(epoch + 1, i / args.accumulate_step + 1))
                if is_master:
                    if is_mp:
                        recorder.save(model.module, "model_cur.bin")
                    else:
                        recorder.save(model, "model_cur.bin")
                    recorder.save(optimizer, "optimizer_cur.bin")
                    recorder.print("val score: %.6f"%(1 - loss))


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=12355)
    parser.add_argument("--model_pt", default=None, type=str)
    parser.add_argument("--model_name", default=None, type=str)
    args = parser.parse_args()
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:    
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)