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

cuniv_stats= 0
#-percentage of improvements for top 5%largest snippets for python and java veso/unixcoder in MRR
def rq_analysis1(ranks, ranks0, query_dataset, topk=5):#ranks is alredy 1/rank
    # Open the file in write mode ('w') and get the file object
    list_queries = []
    with open('output_an1.txt', 'w') as output_file:
        cequal = 0
        cup = 0
        cdown = 0
        for r_idx in range(len(ranks)):
            if ranks[r_idx] > ranks0[r_idx]:
                cup += 1
                #list_queries_up += [(query_dataset.examples[r_idx].code.count('\n'), r_idx)]
            elif ranks[r_idx] < ranks0[r_idx]:
                cdown += 1
                #list_queries_down += [(query_dataset.examples[r_idx].code.count('\n'), r_idx)]
            else:
                cequal += 1
            list_queries += [(query_dataset.examples[r_idx].code.count('\n'), r_idx)]
        list_queries = sorted(list_queries, key=lambda x: (x[0], x[1]))[::-1]
        
        # Write the output to the file
        #output_file.write('begin rq analysis {}\n'.format(len(ranks)))
        #output_file.write("upgraded {}\n".format(len(list_queries)))
        #output_file.write('downgraded{}\n'.format(len(list_queries_down)))
        #output_file.write('equal {}\n'.format(cequal))
        print('begin rq analysis ', len(list_queries))
        print("upgraded ",  cup)
        print('downgraded', cdown)
        print('equal ', cequal)
        N = len(list_queries)
        print(f"{len(ranks)} and {N}")
        csize = 0
        improv = 0
        for size, r_idx in list_queries:
            csize += 1
            if csize == 1:
                output_file.write('begin rq analysis 1 {} {}\n'.format(r_idx, size))
            #output_file.write('Idx {} % {:.2f}%\n'.format(csize, (csize * 100) / N))
            if (csize / N) * 100 >= topk:
                output_file.write('{}\n code\n'.format(query_dataset.examples[r_idx].url))
                print('End llegamos', improv/csize )
                output_file.write('{} end rq analysis 1\n'.format(size))
                break
            delta = ranks[r_idx] - ranks0[r_idx]
            improv += delta
            if ranks[r_idx] == 0 or ranks0[r_idx] == 0:
                continue#we are not going to look at ranks with zero!!
        improv /= csize
        output_file.write('=============\n\n')

    return improv


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
    logger.info("***** Running evaluation *****")
    logger.info("New  Num queries = %d", len(newquery_dataset))
    logger.info("New  Num codes = %d", len(newcode_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    code_vecs, nl_vecs = common.get_vecs(query_dataloader, code_dataloader, model,  args)
    scores0 = np.matmul(nl_vecs, code_vecs.T)
    scores0, nl_urls0, code_urls0 = common.simplify_max(scores0, query_dataset, code_dataset, code_dataset, True)
    ranks0, _ = common.get_mrr2(scores0, nl_urls0, code_urls0)
    print(code_vecs.shape, nl_vecs.shape, "ahapes of nl and code\n")
    code_vecs2, _ = common.get_vecs(newquery_dataloader, newcode_dataloader, model, args)
    code_vecs = code_vecs2
    print(code_vecs.shape, nl_vecs.shape, " final shapes of nl and code\n")

    if np.isnan(np.min(nl_vecs)):
        print("ERROR in nl vecs!!!!")
   
    scores = np.matmul(nl_vecs, code_vecs.T)

    logger.info("end nl_vecs and code_vecs ")
    model.eval()    
    
    code_vecs_normalized = np.allclose(np.linalg.norm(code_vecs, axis=1), 1)
    nl_vecs_normalized = np.allclose(np.linalg.norm(nl_vecs, axis=1), 1)
    
    print("Are code_vecs normalized?", code_vecs_normalized)
    print("Are nl_vecs normalized?", nl_vecs_normalized,'\n')
    print("init score shape with ", scores.shape)

    scores, nl_urls, code_urls = commonsw.simplify_max(scores,query_dataset, code_dataset, newcode_dataset)
    ranks, aux_recall = common.get_mrr2(scores, nl_urls, code_urls)   

    print("unixcoder eval_mrr2", float(np.mean(ranks)) )
    all_prob = min(len(nl_urls), len(code_urls))
    rank_prob = 10
    prob_thrs = 0.99
    tot_recall = 0
    
    lst_ana1_topk = [5, 10, 20, 30, 100]
    print('Improvement over the % largest snippets\n')
    for topk in lst_ana1_topk:
        aavg_delta_position = rq_analysis1(ranks, ranks0, query_dataset, topk)
        print("Improvement at %",topk,' ', aavg_delta_position, end='\n')
    print('\n')
    
    for dx in range(len(aux_recall)):
        tot_recall += aux_recall[dx]
        if dx < rank_prob:
            print("Pos-R@", dx, ' ', tot_recall /all_prob)
        if tot_recall/all_prob >= prob_thrs:
            print("Pos-R@", dx, ' ', tot_recall /all_prob)
            break
    print("unixcoder eval_mrr", float(np.mean(ranks)) )
    result = {
        "eval_mrr":float(np.mean(ranks))
    }
    return result

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