import os
from os.path import exists, join
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_data_path(mode, encoder):
    paths = {}
    if mode == 'train':
        paths['train'] = 'data/train_CNNDM_oracle_' + encoder + '.jsonl'
        paths['val']   = 'data/val_CNNDM_' + encoder + '.jsonl'
    else:
        # paths['test']  = 'data/test_CNNDM_bart' + '.jsonl'
        paths['test']  = 'data/test_CNNDM_random_' + encoder + '.jsonl'
    return paths

def get_result_path(save_path, cur_model):
    # result_path = join(save_path, '../result')
    # result_path = join(save_path, '../result')
    result_path = "./"
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'candidate')
    ref_path = join(model_path, 'reference')
    if not exists(dec_path):
        os.makedirs(dec_path)
    if not exists(ref_path):
        os.makedirs(ref_path)
    return dec_path, ref_path


class Recorder():
    def __init__(self, id, log=True):
        self.log = log
        now = datetime.now()
        date = now.strftime("%y-%m-%d")
        self.dir = f"./cache/{date}-{id}"
        if self.log:
            os.mkdir(self.dir)
            self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.writer = SummaryWriter(os.path.join(self.dir, "log"), flush_secs=60)
        
    def write_config(self, args, models, name):
        if self.log:
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                print(name, file=f)
                print(args, file=f)
                print(file=f)
                for (i, x) in enumerate(models):
                    print(x, file=f)
                    print(file=f)
        print(args)
        print()
        for (i, x) in enumerate(models):
            print(x)
            print()

    def print(self, x=None):
        if x is not None:
            print(x)
        else:
            print()
        if self.log:
            if x is not None:
                print(x, file=self.f)
            else:
                print(file=self.f)

    def plot(self, tag, values, step):
        if self.log:
            self.writer.add_scalars(tag, values, step)


    def __del__(self):
        if self.log:
            self.f.close()
            self.writer.close()

    def save(self, model, name):
        if self.log:
            torch.save(model.state_dict(), os.path.join(self.dir, name))