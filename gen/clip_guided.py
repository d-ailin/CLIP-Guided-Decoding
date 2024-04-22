import torch
from lib.data_utils import *
from lib.utils import *
from PIL import Image

from lib.clip_utils import CLIPModel

from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import copy
TERMINAL_TOKENS = ['.', '?', '!']

def check_if_terminal(sentence):
    return any([token in sentence for token in TERMINAL_TOKENS])

def check_if_termindal_w_tokens(input_ids, tokenizer):
    input_ids = input_ids.tolist()
    terminal_ids = [tokenizer.encode(token)[-1] for token in TERMINAL_TOKENS]
    
    return any([t_id in input_ids for t_id in terminal_ids])

def is_end(output_ids, tokenizer):
    '''
    output_ids: tensor, (seq_len)
    '''
    if tokenizer.eos_token_id in output_ids:
        return True
    
def get_rank_scores(x, args):
    
    alpha = args.get('scoring', {}).get('alpha', 0)
    beta = args.get('scoring', {}).get('beta', 1)
    prob_type = args.get('scoring', {}).get('prob_type', 'sent_mean') # sent_sum_log, sum_log, lennorm_sum_log
    
    seq_probs = x['seq_prob_scores']
    all_sent_probs = x['sent_prob_scores']
    if prob_type == 'sent_mean':
        prob_scores = []
        for sent_probs in all_sent_probs:
            prob_scores.append(np.mean(sent_probs))
        prob_score = np.mean(prob_scores)
    elif prob_type == 'sent_sum_log':
        prob_scores = []
        for sent_probs in all_sent_probs:
            prob_scores.append(np.sum(np.log(sent_probs)))
        prob_score = np.mean(prob_scores)
    elif prob_type == 'sum_log':
        prob_score = np.sum(np.log(seq_probs))
    elif prob_type == 'lennorm_sum_log':
        prob_score = np.sum(np.log(seq_probs))/len(seq_probs)
    
    clip_score = np.mean(x['clip_scores']) # it is in sentence level
    
    score = beta*clip_score + alpha*prob_score
    
    return score

def prepare_sampling_params(args):
    model_name = args['model_name']
    algo_name = args['algo_name']
    sampling_params = args['sampling_params']
    
    final_sampling_params = copy.deepcopy(sampling_params)
    
    return final_sampling_params

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, init_len, tokenizer, max_new_tokens=250, model_name='llava_v1_5'):
        self.init_len = init_len
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        
        self.model_name = model_name
        
    def __call__(self, input_ids, scores, **kwargs):
        tokenizer = self.tokenizer
        if input_ids.shape[1] <= self.init_len + 1:
            return False
        
        if input_ids.shape[1] - self.init_len > self.max_new_tokens:
            return True
        
        if self.model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:            
            decoded_input_ids = input_ids.clone()

            # modify the input_ids's padding token to be 32000
            input_ids[input_ids == -1] = tokenizer.pad_token_id # take it as pad token
            sentences = tokenizer.batch_decode(input_ids[:, self.init_len:], skip_special_tokens=True)

        else:
            sentences = tokenizer.batch_decode(input_ids[:, self.init_len:], skip_special_tokens=True)
        # print('sentences', sentences)
        formatted_sentences = [sentence.replace('</s>', '').replace('<unk>', '') for sentence in sentences]
        non_empty_sentences = [sentence for sentence in sentences if len(sentence) > 0] 
        if all([check_if_terminal(sentence) for sentence in non_empty_sentences]):
            return True

        
        return False

from model_utils.generate_wrapper import Wrapper
def _generate_original(model, tokenizer, image_processor, question, image, args=None,  device='cuda', spec='normal', clip_scorer=None, **kwargs):
    
    model_name = args['model_name']
    
    sampling_params = prepare_sampling_params(args)
    
    generate_wrapper = Wrapper(model, args)
    
    if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
        
        from model_utils.blip2_vicuna_instruct import rewrite_blip2_generate_func, prepare_input as prepare_input_blip2
        image_tensor = image_processor(image.convert('RGB')).unsqueeze(0).to(device)
        
        input_obj = prepare_input_blip2(model, question, image_tensor)
        input_ids = input_obj['input_ids'].to(device)
        input_text = question
        new_input_text = input_text
                
        generate_wrapper.add_cache({
            'input_obj': input_obj
        })
        
    
    elif model_name == 'llava_v1_5':
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        conv_mode = LLAVA_CONV_MODE
        
        
        conv = conv_templates[conv_mode].copy()
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
        if spec == 'w_constimg[0]':
            image_tensor = torch.zeros_like(image_tensor)
        elif spec == 'w_constimg[1]':
            image_tensor = torch.ones_like(image_tensor)
            
    elif model_name == 'mplug_owl2':
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)


    new_input_ids = input_ids
    init_len = input_ids.shape[1]
    last_index = input_ids.shape[1]
    
    candidate_list = [
        (new_input_ids, [0], init_len, last_index, [])
    ]
    candidate_list = [
        {
            'input_ids': new_input_ids,
            'root_record': [0],
            'init_len': init_len,
            'last_index': last_index,
            'clip_scores': [],
            'full_context': '',
            'prob_scores': [],
            'seq_prob_scores': [],
            'sent_prob_scores': [],
        }
    ]
    if args.get('scoring', {}).get('max_cand_num', 0) > 0:
        max_candidate_num = args['scoring']['max_cand_num']
    else:
        max_candidate_num = sampling_params['num_return_sequences']
    
    final_res_list = []
    
    sub_sampling_params = copy.deepcopy(sampling_params)
    
    
    while True:
        
        select_list = []
        for candidate in candidate_list:
            cand_input_ids = candidate['input_ids']
            root_record = candidate['root_record']
            cand_init_len = candidate['init_len']
            cand_last_index = candidate['last_index']
            cand_clip_scores = candidate['clip_scores']
            cand_full_context = candidate['full_context']
            cand_prob_scores = candidate['prob_scores']
            cand_seq_prob_scores = candidate['seq_prob_scores']
            cand_sent_prob_scores = candidate['sent_prob_scores']
            
            end_sentence_criteria = CustomStoppingCriteria(init_len=cand_init_len, tokenizer=tokenizer, max_new_tokens=sampling_params['max_new_tokens'], model_name=model_name)
            criterias = StoppingCriteriaList([end_sentence_criteria])
            
            with torch.inference_mode():
                
                if kwargs.get('verbose', False):
                    print('cand_input_ids', cand_input_ids.shape, cand_input_ids)

                outputs = generate_wrapper.generate(
                    input_ids=cand_input_ids,
                    images=image_tensor.unsqueeze(0).half().to(device),
                    return_prob=True,
                    stopping_criteria=criterias,
                    **sub_sampling_params
                )
                
                output_ids = outputs['output_ids']
                output_token_probs = outputs['output_token_probs']
                
            
            if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
                decoded_output_ids = output_ids.clone()
                decoded_output_ids[decoded_output_ids == 0] = 2
            else:
                decoded_output_ids = output_ids
            
            sentences = tokenizer.batch_decode(decoded_output_ids[:, cand_last_index:], skip_special_tokens=True)
            
            
            one_sentences = []
            for i, sentence in enumerate(sentences):

                token_indexs = []
                for token in TERMINAL_TOKENS:
                    if token in sentence:
                        token_indexs.append(sentence.index(token))
                if len(token_indexs) > 0:
                    # include the end token, +1
                    sentence = sentence[:min(token_indexs)+1].strip()
                
                # if duplicate, then skip
                if one_sentences.count(sentence) > 0:
                    if kwargs.get('verbose', False):
                        print('skip duplicate sentence', sentence)
                    continue
                one_sentences.append(sentence)
                
                
                start_index = cand_last_index # start from the next token, not include the previous end token
                last_index = start_index
                for j, token in enumerate(output_ids[i, start_index:]):
                    if check_if_terminal(tokenizer.decode(token, skip_special_tokens=True)):
                        last_index += j
                        break
                    elif j == len(output_ids[i, start_index:]) - 1:
                        # other wise use the last one
                        last_index += j
                last_index += 1
                
                
                if len(sentence) == 0:
                    new_clip_scores = cand_clip_scores
                    new_prob_scores = cand_prob_scores
                    new_seq_prob_scores = cand_seq_prob_scores
                    new_sent_prob_scores = cand_sent_prob_scores
                else:
                    clip_score = clip_scorer.get_clip_score(sentence, image)
                    new_clip_scores = cand_clip_scores + [clip_score]
                    
                    trim_cand_last_index = cand_last_index - cand_input_ids.shape[1]
                    trim_last_index = last_index - cand_input_ids.shape[1]
                    seq_prob_scores = output_token_probs[i, trim_cand_last_index:trim_last_index]
                    
                    prob_scores = seq_prob_scores.mean().item()
                    
                    seq_prob_scores = seq_prob_scores.tolist()
                    
                    
                    if kwargs.get('verbose', False):
                        print('sentence', sentence, clip_score, prob_scores, output_token_probs[i, trim_cand_last_index:trim_last_index], output_ids[i, cand_last_index:])
                    
                    new_prob_scores = cand_prob_scores + [prob_scores]
                    
                    new_seq_prob_scores = cand_seq_prob_scores + seq_prob_scores
                    new_sent_prob_scores = cand_sent_prob_scores + [seq_prob_scores]
                    
                
                if kwargs.get('verbose', False):
                    print('checking is_end output_ids:', output_ids[i, start_index:last_index])
                    
                if is_end(output_ids[i, start_index:last_index], tokenizer):
                    
                    final_res_list.append({
                        'output_ids': output_ids[i, :last_index].unsqueeze(0),
                        'root_record': root_record + [i],
                        'clip_scores': new_clip_scores,
                        'prob_scores': new_prob_scores,
                        'seq_prob_scores': new_seq_prob_scores,
                        'sent_prob_scores': new_sent_prob_scores
                    })
                    if kwargs.get('verbose', False):
                        print('push to final_res_list', sentence)
                    # not append to select_list
                    continue

                
                select_list.append({
                    'input_ids': output_ids[i, :last_index].unsqueeze(0),
                    'root_record': root_record+[i],
                    'init_len': output_ids[i, :last_index].unsqueeze(0).shape[1],
                    'last_index': last_index,
                    'clip_scores': new_clip_scores,
                    'sentence': sentence,
                    'full_context': cand_full_context + sentence,
                    'prob_scores': new_prob_scores,
                    'seq_prob_scores': new_seq_prob_scores,
                    'sent_prob_scores': new_sent_prob_scores,
                })
            
        sorted_select_list = sorted(select_list, key=lambda x: get_rank_scores(x, args), reverse=True)
        
        if kwargs.get('verbose', False):
            print('sorted_select_list', len(sorted_select_list))
            for candidate in sorted_select_list:
                print(candidate['full_context'],  np.mean(candidate['clip_scores']), np.mean(candidate['prob_scores']), candidate['root_record'], candidate['clip_scores'], candidate['prob_scores'])
                print('-------------------')
            print()
        
        candidate_list = sorted_select_list[:max_candidate_num]                
        
        if kwargs.get('verbose', False):
            print('candidate_list', len(candidate_list))
            for candidate in candidate_list:
                print(candidate['full_context'],  np.mean(candidate['clip_scores']), np.mean(candidate['prob_scores']), candidate['root_record'], candidate['clip_scores'], candidate['prob_scores'])
                print('-------------------')
            
        # if all candidate_list exceed max_new_tokens, then stop
        if len(candidate_list) > 0 and all([candidate['init_len'] - input_ids.shape[1] > sampling_params['max_new_tokens'] for candidate in candidate_list]):
            # final_res_list.extend([(candidate['input_ids'], candidate['root_record']) for candidate in candidate_list])
            
            for candidate in candidate_list:
                final_res_list.append({
                    'output_ids': candidate['input_ids'],
                    'root_record': candidate['root_record'],
                    'clip_scores': candidate['clip_scores'],
                    'prob_scores': candidate['prob_scores'],
                    'seq_prob_scores': candidate['seq_prob_scores'],
                    'sent_prob_scores': candidate['sent_prob_scores'],
                })
            
            if kwargs.get('verbose', False):
                print('exceed max_new_tokens')
            break
        
        # if final_res_list exceed num_return_sequences, then stop
        if len(final_res_list) >= sampling_params['num_return_sequences']:
            if kwargs.get('verbose', False):
                print('exceed num_return_sequences')
            break
        
        if len(candidate_list) == 0:
            if kwargs.get('verbose', False):
                print('candidate_list is empty')
            break

    outputs = []
    rank_scores = []
    
    for res in final_res_list:
        output_ids = res['output_ids']
        rank_scores.append( get_rank_scores(res, args))
    
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
            decoded_output_ids = output_ids.clone()
            decoded_output_ids[decoded_output_ids == 0] = 2
        else:
            decoded_output_ids = output_ids
        _outputs = tokenizer.batch_decode(decoded_output_ids[:, input_token_len:], skip_special_tokens=True)
        _outputs = [output.strip() for output in _outputs]
        # outputs = []
        if model_name == 'llava_v1_5':
            for _output in _outputs:
                if _output.endswith(stop_str):
                    _output = _output[:-len(stop_str)]
                _output = _output.strip()
                outputs.append(_output)
        elif model_name == 'mplug_owl2':
            outputs.extend(_outputs)
        elif model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
            outputs.extend(_outputs)
    
    # sort by clip scores by descending order
    sorted_idx = np.argsort(rank_scores)[::-1]
    sorted_outputs = [outputs[idx] for idx in sorted_idx]
    
    if kwargs.get('verbose', False):
        print('sorted_outputs')
        # output sentences and scores
        for i, output in enumerate(sorted_outputs):
            print(output, rank_scores[sorted_idx[i]], final_res_list[sorted_idx[i]]['clip_scores'], final_res_list[sorted_idx[i]]['prob_scores'])
            print('-------------------')
    
    return sorted_outputs


def generate_w_clip(model, tokenizer, image_processor, question, image, args=None,  device='cuda', spec='normal', clip_scorer=None, **kwargs):
    
    outputs = _generate_original(model, tokenizer, image_processor, question, image, args=args,  device=device, spec=spec, clip_scorer=clip_scorer, **kwargs)

    if kwargs.get('verbose', False):
        print('outputs')
        for output in outputs:
            print(output)
            print('-------------------')
        # print('outputs', outputs)
    
    # just return the best one
    return [outputs[0]]


if __name__ == '__main__':
    from lib.utils import set_seed
    set_seed(0)
    
    model_name = 'llava_v1_5'
    model_type = '7b'

    # model_name = 'blip2_vicuna_instruct'
    # model_type = 'vicuna7b'
    
    # model_name = 'mplug_owl2'
    # model_type = 'llama2-7b'
        
    
    device = 'cuda:0'
    print('begin loading model')
    model, vis_processor, tokenizer = load_model(model_name=model_name, model_type=model_type, device=device)
    print('end loading model')

    image_id = '159030'
    image_path = '/data/ailin/coco/val2014/COCO_val2014_000000{}.jpg'.format(image_id)
    image = Image.open(image_path).convert("RGB")
    

    if model_name == 'llava_v1_5':
        text = "<image>\nDescribe this image in detail."
    elif model_name == 'blip2_vicuna_instruct':
        text = "Describe this image in detail."
    elif model_name == 'mplug_owl2':
        text = "<|image|>Describe this image in detail.\n"
    

    
    args = {
        'model_name': model_name,
        'model_type': model_type,
        'algo_name': 'sentence_clip',
        'sampling_params': {
            'temperature': 0.2,
            'top_p': 1,
            'top_k': 5, # added
            'num_beams': 1,
            'do_sample': True,
            'num_return_sequences': 3,
            'max_new_tokens': 500,
        },
        'scoring': {
            'alpha': 0.1, # prob
            'beta': 1, # clip
            'prob_type': 'lennorm_sum_log'
        }
    }


    clip_scorer = CLIPModel(device=device)

    all_generated_str = sampling_sentence_w_clip(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
                                                                 return_max_probs=True, return_clip_scores=True, clip_scorer=clip_scorer)
    
    for generated_str in zip(all_generated_str):
        print(generated_str)
