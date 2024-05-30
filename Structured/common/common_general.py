from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import torch
import json
import numpy as np
import random
import os
import sys

sys.path.insert(1,'../../../')
from common.model import Model

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initarguments(parser):
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--block_size", default=256, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")


    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=15,
                        help="random seed for initialization")
    parser.add_argument('--line_sw', type=int, default=42,
                        help="random seed for initialization")
    return parser

def LoadModel(args, logging, logger):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model) 
    return args, tokenizer, model, config


def get_vecs(query_dataloader, code_dataloader, model, args):
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 
    
    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    return code_vecs, nl_vecs

def get_predictions(query_dataset, sort_ids, args):
    indexs=[]
    urls=[]
    for example in query_dataset.examples:
        indexs.append(example.idx)
        urls.append(example.url)
    with open(os.path.join(args.output_dir,"./predictions.jsonl"),'w') as f:
        for index,url,sort_id in zip(indexs,urls,sort_ids):
            js={}
            js['url']=url
            js['answers']=[]
            urls = {}
            for idx in sort_id:
                indxx = indexs[int(idx)]
                if indxx in urls:
                    continue
                urls[indxx] = True
                js['answers'].append(indexs[int(idx)])
            f.write(json.dumps(js)+'\n')

def get_mrr(scores, nl_urls, code_urls):#doesn't work
    ranks = []
    offset = 2000
    aux_recall = [0] *  (2 * offset)
    
    for i, url in enumerate(nl_urls):
        rank = 0
        find = False
        scorei = scores[:,i]
        sort_id = np.argsort(scorei)[::-1]
        for idx in sort_id[:offset]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        aux_recall[rank] += 1
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    return ranks, aux_recall

def get_mrr2(scores, nl_urls, code_urls): #nl_urls > code_urls
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    ranks = []
    offset = 2000
    aux_recall = [0] *  (2 * offset)
    
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:offset]:
            if find is False:
                rank += 1
            if idx < len(code_urls) and code_urls[idx] == url:
                find = True
        aux_recall[rank] += 1
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    return ranks, aux_recall

