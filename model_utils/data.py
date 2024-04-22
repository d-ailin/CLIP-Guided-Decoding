
def get_last_sentence_indexs(input_ids, tokenizer, terminal_tokens=['.']):
    '''
    input_ids: tensor, (num_samples, seq_len)
    '''
    last_sentence_indexs = []
    for i in range(input_ids.shape[0]):
        last_sentence_index = 0
        for j in range(input_ids.shape[1]):
            token = tokenizer.decode(input_ids[i, j])
            if token in terminal_tokens:
                last_sentence_index = j
        last_sentence_indexs.append(last_sentence_index)
    return last_sentence_indexs

import torch
def get_all_sentence_indexs(input_ids, tokenizer, terminal_tokens=['.']):
    '''
    input_ids: tensor, (num_samples, seq_len)
    '''
    all_sentence_indexs = []
    for i in range(input_ids.shape[0]):
        sentence_indexs = []
        for j in range(input_ids.shape[1]):
            token = tokenizer.decode(input_ids[i, j], skip_special_tokens=True)
            if token in terminal_tokens:
                sentence_indexs.append(j)
        all_sentence_indexs.append(torch.tensor(sentence_indexs).int().to(input_ids.device))
    
    
    return all_sentence_indexs