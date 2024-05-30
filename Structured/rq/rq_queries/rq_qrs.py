# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import sys
sys.path.insert(1,'../../../')
import common.common_general as common
import common.common_input as commoninput
import common.common_sw as commonsw
#import common.common_input_space as commoninput2
import argparse
import logging
import os
import pickle
import random
import torch
import json
import copy
import numpy as np
import common.RemoveComents as rmcm

from common.model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

device = "cuda"
univcsize=0

cuniv_stats= 0


def get_mrr3(scores, nl_urls, code_urls): #nl_urls > code_urls
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    offset = 10
    sim = []
    idxi = -1
    print(len(sort_ids), len(nl_urls), ' sizes')
    for url, sort_id in zip(nl_urls, sort_ids):
        idxi += 1
        rank = 0
        find = False
        cfirst = 1
        for idx in sort_id[:offset]:
            idxj = idx
            if find is False:
                rank += 1
            if idxi != idxj and cfirst>=0:
                cfirst -= 1
                sim += [scores[idxi, idxj]]
            if idx < len(code_urls) and code_urls[idx] == url:
                find = True
                if idxj != idxi:
                    print("epa error idxi != idxj")
    sim.sort()
    return sim
def evaluate(args, model, tokenizer,file_name, range_windows,eval_when_training=False):
    query_dataset = commoninput.TextDataset(tokenizer, args, file_name)
    code_dataset = query_dataset
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    code_vecs, nl_vecs = common.get_vecs(query_dataloader, code_dataloader, model,  args)
    print(code_vecs.shape, nl_vecs.shape, "shapes of nl and code\n")

    print(code_vecs.shape, nl_vecs.shape, " final shapes of nl and code\n")

    if np.isnan(np.min(nl_vecs)):
        print("ERROR in nl vecs!!!!")
   
    scores = np.matmul(nl_vecs, code_vecs.T)
    
    code_vecs_normalized = np.allclose(np.linalg.norm(code_vecs, axis=1), 1)
    nl_vecs_normalized = np.allclose(np.linalg.norm(nl_vecs, axis=1), 1)
    
    print("Are code_vecs normalized?", code_vecs_normalized)
    print("Are nl_vecs normalized?", nl_vecs_normalized,'\n')
    code_urls = [ex.url for ex in code_dataset.examples ]
    nl_urls = [ex.url for ex in query_dataset.examples ]
    sim = get_mrr3(scores, nl_urls, code_urls)
    indices = range(len(sim))
    print(len(sim), 'len sim')
    
    
    plt.hist(sim, bins=10)

    plt.xlabel("Frequency")
    plt.ylabel("Cos-similarity")
    plt.show()
    plt.savefig("sorted_values.png")

    return {}    

def main():
    parser = argparse.ArgumentParser()
    parser =common.initarguments(parser)
    args = parser.parse_args()
    args, tokenizer, model, config = common.LoadModel(args, logging, logger)
    if args.do_train:
        common.train(args, model, tokenizer)
    print("Debug\n")
    print(args.model_name_or_path)
    print(args.block_size)
    print("Fin\n")
    results = {}
    if args.do_eval:
        common.do_eval(args, model, tokenizer,logger)
    
    lst_lines = [args.line_sw]
    
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            print("loaded")
        model.to(args.device)
        logger.info("***** Eval results *****")
        result = evaluate(args, model, tokenizer,args.test_data_file, lst_lines)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
        import gc
        gc.collect()
        import psutil
        process = psutil.Process()
        print(process.memory_info().rss)
        logger.info("******** Try again ********")

if __name__ == "__main__":
    main()