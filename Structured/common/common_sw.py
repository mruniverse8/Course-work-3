import sys
import numpy as np

sys.path.insert(1,'../../../')

import common.common_input_space as commoninput


def expand(codes, tokenizer, limit_lines, limit_size,stride=3,MAX_LINES=30):#Stride will be universally put to 5
    lst_code = []
    per_id = codes.per_id
    distance_endl = 14
    code = codes.code
    ant = -1
    act = 0
    lst_pos_endl = [0]
    for chr in code:
        if chr == ' ' and act - ant >= distance_endl:
            lst_pos_endl += [act]
            ant = act
        act += 1
    
    lst_pos_endl += [len(code)]
    #debuged
    if (len(lst_pos_endl) < MAX_LINES):
        return [codes]
    limit_lines = min(limit_lines, len(lst_pos_endl) - 1)
    #limit_lines = min(limit_lines, stride) #may be update this
    for i in range(0, len(lst_pos_endl), stride):
        idx_initial = lst_pos_endl[i]
        if (i + limit_lines >= len(lst_pos_endl)):
            break
        idx_final  = lst_pos_endl[i + limit_lines]
        chunk_code = code[idx_initial:idx_final]
        _, code_ids = commoninput.conver_code_to_id(chunk_code, tokenizer, limit_size)
        lst_code += [commoninput.InputFeatures(None, code_ids, None, codes.nl_ids ,codes.url, codes.idx, chunk_code, None, per_id, limit_lines)]

    if len(lst_code) == 0:
        print("Error in lst code")
        exit(0)
    #it may be the case where we don't consider that we put two codes at the same time
    return lst_code

def AugmentData(code_dataset, tokenizer, range_windows, limit_size):
    newcode_dataset = []
    new_code_ex = []
    for limit_lines in range_windows:
        cant_codes = 0
        for code in code_dataset.examples:
            code = expand(code, tokenizer, limit_lines, limit_size)
            if len(code) > 1:
                cant_codes += 1
            new_code_ex += code
        print('begin :', limit_lines)
        print('end with % toched codes ', cant_codes/ len(code_dataset.examples))
    print(len(new_code_ex), 'Len code in augmentation')
    newcode_dataset = commoninput.TextDataset(None, None, None, aux_examples=new_code_ex)
    print("Verify ", len(newcode_dataset))
    return newcode_dataset
#newcode has all the dataset
def simplify_max(scores, query_dataset, code_dataset, newcode_dataset,  nomax=False):
    nl_urls = [example.idx for example in query_dataset.examples]
    code_urls = []
    mp_pos = {}

    for example in code_dataset.examples:
        if example.per_id not in mp_pos:
            mp_pos[example.per_id] = []
        mp_pos[example.per_id].append(len(code_urls))
        code_urls.append(example.idx)
    if nomax == True:
        print("simplify max skipped")
        return scores, nl_urls, code_urls
    code_urls2 = code_urls
    size_code_urls = len(code_urls)
    scores_f = np.zeros((len(nl_urls), len(code_urls)))
    code_urls = []
    print(scores_f.shape)
    for example in newcode_dataset.examples:
        if example.per_id not in mp_pos:
            print("error it should be added")
            mp_pos[example.per_id] = []
        mp_pos[example.per_id].append(len(code_urls))
        code_urls.append(example.idx) # Augmentations

    #return scores, nl_urls, code_urls
    for example in code_dataset.examples:
        first_pos = mp_pos[example.per_id][0]
        if len(mp_pos[example.per_id])> 1:
            score_max = np.max(scores[:, mp_pos[example.per_id][1:]], axis=1)
        else:
            print("Error of code_urls")
            exit(0)
        score_max = np.squeeze(score_max)
        scores_f[:, first_pos] = score_max
    
    return scores_f, nl_urls, code_urls2