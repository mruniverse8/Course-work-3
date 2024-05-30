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
import common.common_input_space as commoninput
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
#import common.RemoveComents as rmcm

from common.model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset


logger = logging.getLogger(__name__)

device = "cuda"
univcsize=0

def rq_analysis4(code_dataset, query_dataset , code_vecs, nl_vecs, lst_range):
    #this doesn't work if code_dataset != query_dataset
    print("MMR in range groups:")
    scores = np.matmul(code_vecs, nl_vecs.T)
   
    for idxqry in range(len(lst_range)):
        li, ri = lst_range[idxqry]
        lst_snipt = []
        lst_snipt2 = []
        mapa = {}
        for idx in range(len(code_dataset.examples)):
            code_dataseti = code_dataset.examples[idx]
            lines = code_dataseti.code.count('\n')
            mapa[code_dataseti.url] =code_dataseti.code #added for java
            if lines >= li and lines <= ri:
                lst_snipt.append(idx)
        
        for idx in range(len(query_dataset.examples)):
            query_dataseti = query_dataset.examples[idx]
            lines = mapa[query_dataseti.url].count('\n')
            if lines >= li and lines <= ri:
                lst_snipt2.append(idx)
        
        nl_urls = [example.url for example in query_dataset.examples]
        group_nl_urls =[query_dataset.examples[idx].url for idx in lst_snipt2]
        group_code_urls = [code_dataset.examples[idx].url for idx in lst_snipt]
        group_scores = np.take(scores, lst_snipt, axis=0)
        group_ranks, _ = common.get_mrr2(group_scores, nl_urls, group_code_urls)
        print(f"Range = [{li}, {ri}] lines of code")
        print(f"list in codes {len(lst_snipt)} and list in queries {len(lst_snipt2)}")
        print(f"MMR-all queries ={float(np.mean(group_ranks))}")
        group_scores = np.take(group_scores, lst_snipt2, axis=1)
        group_ranks, _ = common.get_mrr2(group_scores, group_nl_urls, group_code_urls)
        print(f"MMR-queries ={float(np.mean(group_ranks))}\n")
cuniv_stats= 0
def evaluate(args, model, tokenizer,file_name, range_windows,eval_when_training=False):
    query_dataset = commoninput.TextDataset(tokenizer, args, file_name)
    code_dataset = commoninput.TextDataset(tokenizer, args, args.codebase_file)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    newquery_dataset, newcode_dataset = query_dataset, commonsw.AugmentData(code_dataset, tokenizer, range_windows, args.block_size)
    logger.info(" check  new  queries = %d", len(newquery_dataset))
    logger.info(" check  new  codes = %d", len(newcode_dataset))    
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    
    
    newquery_sampler = SequentialSampler(newquery_dataset)
    newquery_dataloader = DataLoader(newquery_dataset, sampler=newquery_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    newcode_sampler = SequentialSampler(newcode_dataset)
    newcode_dataloader = DataLoader(newcode_dataset, sampler=newcode_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # Eval!
    logger.info("New  Num queries = %d", len(newquery_dataset))
    logger.info("New  Num codes = %d", len(newcode_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    range_windows = [(1, 2), (3, 5), (5, 15), (15, 70)]
    code_vecs, nl_vecs = common.get_vecs(query_dataloader, code_dataloader, model,  args)
    code_vecs2, _ = common.get_vecs(newquery_dataloader, newcode_dataloader, model,  args)
    print("Begin rq4:\n")
    print("Expected Similarity between initial data\n")
    rq_analysis4(code_dataset, query_dataset, code_vecs, nl_vecs, range_windows)
    print("Expected Similarity between final data\n\n\n")
    rq_analysis4(newcode_dataset, query_dataset, code_vecs2, nl_vecs, range_windows)
    
    code_vecs = code_vecs2 
    if np.isnan(np.min(nl_vecs)):
        print("ERROR in nl vecs!!!!")
    
    scores = np.matmul(nl_vecs, code_vecs.T)
    model.eval()
    
    scores, nl_urls, code_urls = commonsw.simplify_max(scores, query_dataset, code_dataset, newcode_dataset, False)
    ranks,_ = common.get_mrr2(scores, nl_urls, code_urls)
    print("eval_mrr", float(np.mean(ranks)) )

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
            checkpoint_prefix = 'model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
            print("Loaded!!")
        model.to(args.device)
        logger.info("***** Eval results *****")
        result = evaluate(args, model, tokenizer,args.test_data_file, lst_lines)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
        import gc
        gc.collect()
        logger.info("******** Try again ********")

if __name__ == "__main__":
    main()