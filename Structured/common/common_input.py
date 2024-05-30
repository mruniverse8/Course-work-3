from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import json 
import random
import numpy as np
import torch

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 idx,
                 code,
                 nl,
                 per_id=-1,
                 window_size=-1

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.idx=idx
        self.code=code
        self.nl=nl
        self.per_id = per_id
        self.window_size = window_size

def conver_code_to_id(code, tokenizer, block_size):
    code_tokens = tokenizer.tokenize(code)[:block_size-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = block_size- len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    return code_tokens,code_ids

def convert_examples_to_features(js,tokenizer,args, per_id):
     """convert examples to token ids"""
     code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
     #code = js['code']
     code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
     code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
     code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
     padding_length = args.block_size- len(code_ids)
     code_ids += [tokenizer.pad_token_id]*padding_length
     nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
     nl_tokens = tokenizer.tokenize(nl)[:args.block_size-4]
     nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
     nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
     padding_length = args.block_size- len(nl_ids)
     nl_ids += [tokenizer.pad_token_id]*padding_length    
     #code = new_parser(js['code'])
     return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"],
                          js['url'] if "url" in js else js["retrieval_idx"], code, nl, per_id)
 #  Here we modify to accept java code js["idx"] is only in Advtest
class TextDataset(Dataset):

    def __init__(self, tokenizer, args, file_path=None, aux_examples=None):
        if args == None:
            print(len(aux_examples), "entro very")
            self.examples = aux_examples
            return None
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                        js['code'] = js['function']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                print(js)
                for key in js:
                    temp = {}
                    temp['code'] = key
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 
        per_id = 0
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args, per_id))
            per_id += 1
                
    """ if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          """                    
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))

