from nltk.metrics import distance
import pathlib
import torch

def get_save_paths(args):
    
    if args['dataset_name'] == 'nocaps':
        full_dataset_name = 'nocaps/{}'.format(args['domain'])
    else:
        full_dataset_name = args['dataset_name']
        
    path_prefix = '/{}/{}_{}/{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(
        full_dataset_name,
        args['q_type'], args['q_content'],
        args['model_name'], args['model_type'],
        args['algo_name'], args['algo_version'], args['clip_model_name'], args['clip_model_pretrain'], args['tag'], args['seed'], args['test_sample_num']
    )
    
    if args['using_sampling_params']:
        path_prefix += '_sampling={}_topp={}_topk={}_nbeam={}_num={}_t={}'.format(
            args['sampling_params']['do_sample'],
            args['sampling_params']['top_p'],
            args['sampling_params']['top_k'],
            args['sampling_params']['num_beams'],
            args['sampling_params']['num_return_sequences'],
            args['sampling_params']['temperature'],
        )
        if args.get('using_scoring_params', False):
            path_prefix += '_alpha={}_beta={}_probtype={}_maxcand={}'.format(args['scoring']['alpha'], args['scoring']['beta'], args['scoring']['prob_type'], args['scoring']['max_cand_num'])
    
    paths = {
        'prefix': path_prefix,
        'output_path': 'outputs/' + path_prefix + '.json',
        'res_intermediate_path': 'outputs/' + path_prefix + '_res_intermediate.json',
        'res_sent_intermediate_path': 'outputs/' + path_prefix + '_res_sent_intermediate.json',
        'res_path': 'outputs/' + path_prefix + '_res.json',
    }
    
    
    if args.get('eval_seem_labels', False):
        paths['res_intermediate_path'] = paths['res_intermediate_path'].replace('res', 'w_seem_res.json')
        paths['res_path'] = paths['res_path'].replace('res', 'w_seem_res')
    
    if args['task'] == 'eval_mmvet':
        paths['res_intermediate_path'] = paths['res_intermediate_path'].replace('res', 'mmvet_res.json')
    
    # eval vcd and opera output directly
    if args['algo_name'] in ['vcd', 'opera'] or args['tag'] == 'out_baseline':
        new_prefix_path = '/home/ailin/proj/HALC/'
        paths['output_path'] = new_prefix_path + paths['output_path']
    
    # create dir if not exist
    pathlib.Path(paths['output_path']).parent.mkdir(parents=True, exist_ok=True)
    

    return paths

def create_question(dataset_name, q_obj={}, q_type='plain', q_content='select_best_caption'):
    
    
    if q_content == 'select_best_caption':
        q_captions = q_obj['q_captions']

        if q_type == 'plain':
            template = "Which is the best caption for the image among the following captions: {}?"
            formatted_string = ", ".join(f"'{element}'" for element in q_captions)
            q = template.format(formatted_string)
        elif q_type == 'multi_choice':
            template = "Which is the best caption for the image among the following captions: \n{}"
            formatted_string = "\n".join(f"({_i+1}) '{element}'" for _i, element in enumerate(q_captions))
            q = template.format(formatted_string)
        elif q_type == 'multi_choice_ab':
            # from 1-26 to A-Z
            ch_map = {i: chr(i+65) for i in range(26)}
            template = "Which is the best caption for the image among the following captions: \n{}"
            formatted_string = "\n".join(f"({ch_map[_i]}) '{element}'" for _i, element in enumerate(q_captions))
            q = template.format(formatted_string)
    elif q_content == 'caption':
        
        templates = {
            'short': 'Generate a short caption of the image.',
            'brief': 'Provide a brief description of the image.',
            'concise': 'Generate a concise description for the image.',
            'summary': 'Create a short textual summary for the image.',
            'describe': 'Describe this image.',
            'describe_detailed': 'Describe this image in detail.',
        }
        
        q = templates[q_type]
        
    return q



import numpy as np

def turn_token_score_in_word(generated_text, generated_ids, scores, tokenizer=None):
    clean_generated_text = generated_text.strip()
    words = clean_generated_text.split()
    
    tokens = [tokenizer.batch_decode([g_id])[0] for g_id in generated_ids]
    pieces = [tokenizer.sp_model.id_to_piece(g_id.item()) for g_id in generated_ids]

    word_scores = [[]]
    word_i = 0
    for i, t in enumerate(tokens):
        if (pieces[i].startswith('▁') and pieces[i] != '▁' and i > 0) or (i > 2 and '<0x0A>' in tokens[i-1] and '<0x0A>' not in tokens[i]):
            word_i += 1
        elif i > 0 and pieces[i-1] == '▁' and not pieces[i].startswith('▁') and pieces[i] != '<0x0A>':
            word_i += 1
        
        if t == '\n': continue
        if t == ' ': continue
        if '<0x0A>' in t: continue
        if t in words[word_i]:
            if len(word_scores) <= word_i:
                word_scores.append([])
            word_scores[word_i].append(scores[i])
        
    final_word_scores = []
    for i in range(len(word_scores)):        
        # we can also use the first token prob
        final_word_scores.append(word_scores[i][0])
        
    return final_word_scores

from nltk import word_tokenize, pos_tag

def is_last_word_noun(text):
    try:
        word = text.split()[-1]

        if len(word.lower()) == 1:
            return False

        
        word_tokenized = word_tokenize(word)
        
        # Get parts of speech for the word
        pos = pos_tag(word_tokenized)
        
        # Check if the primary part-of-speech tag is a noun type (NN, NNP, NNPS, NNS)
        return pos[0][1] in ['NN', 'NNS', 'NNP', 'NNPS']
    except:
        return False

from pattern.en import singularize
def format_word(w):
    return singularize(w.lower().replace(',', '').replace('.', ''))

def free_gpu_memory():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

import json
def load_mmvet_question(mmvet_path, args=None):
    mmvet_questions = []
    
    mmvet_q_json = json.load(open(mmvet_path, 'r'))
    
    i = 1
    for k, q_obj in mmvet_q_json.items():
        image_id = k
        new_obj = {
            # 'question_id': i,
            # 'image_id': image_id,
            'question_id': k,
            'image_id': q_obj['imagename'].replace('.jpg', '').replace('.png', ''),
            'image': q_obj['imagename'],
            'question': q_obj['question'],
            'label': q_obj['answer'],
        }
        
        mmvet_questions.append(new_obj)
        i += 1
    

    return mmvet_questions
    

from omegaconf import DictConfig, OmegaConf

def save_res_to_json(res, save_path):
    try:
        # if DictConfig: convert to dict
        if res['config']['using_sampling_params'] and isinstance(res['config']['sampling_params'], DictConfig):
            res['config']['sampling_params'] = OmegaConf.to_container(res['config']['sampling_params'])
        if res['config']['using_scoring_params'] and isinstance(res['config']['scoring'], DictConfig):
            res['config']['scoring'] = OmegaConf.to_container(res['config']['scoring'])
    except:
        pass
        
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=4)
    print('saved json res to {}'.format(save_path))


def is_end(token_id, tokenizer=None, args=None):
    if args['model_name'] == 'llava_v1_5':
        return token_id == tokenizer.eos_token_id
    elif args['model_name'] == 'blip2_vicuna_instruct':
        return token_id == 0 or token_id == 2
    elif args['model_name'] == 'mplug_owl2':
        return token_id == tokenizer.eos_token_id
    
    
def is_start_of_a_word(token, tokenizer=None, args=None):
    return tokenizer.sp_model.IdToPiece(token.item()).startswith('\u2581')

def is_end_of_a_sentence(token, tokenizer=None, args=None):
    # return tokenizer.sp_model.IdToPiece(token.item()) == '.'
    return tokenizer.sp_model.IdToPiece(token.item()) in ['.', '?', '!']

def is_end_of_a_statement(token, tokenizer=None, args=None):
    return tokenizer.sp_model.IdToPiece(token.item()) in [',', '.', '?', '!']

def prepare_input(text='', image=None, tokenizer=None, vis_processor=None, device='cuda', args=None, model=None):
    input_obj = {}
    if args['model_name'] == 'llava_v1_5':
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        image_tensor = vis_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).half().to(device)

        input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        input_obj['input_ids'] = input_ids
        input_obj['image_tensor'] = image_tensor
    
    elif args['model_name'] == 'blip2_vicuna_instruct':
        from model_utils.blip2_vicuna_instruct import prepare_input as prepare_input_blip2

        # image = vis_processor["eval"](image.convert('RGB')).unsqueeze(0).to(device)
        image = vis_processor(image.convert('RGB')).unsqueeze(0).to(device)
        input_obj = prepare_input_blip2(model, text, image)
        
        # print('input_obj', input_obj)
    elif 'mplug_owl2' in args['model_name']:
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], vis_processor)
        image_tensor = image_tensor.to(device, dtype=torch.float16)

        # image_tensor = vis_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # image_tensor = image_tensor.unsqueeze(0).to(device)

        input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        input_obj['input_ids'] = input_ids
        input_obj['image_tensor'] = image_tensor
    
    return input_obj

def turn_tokens_to_clip_text(tokens, tokenizer, model=None, args=None):
    # temp_generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    temp_generated_text = decode_generated_ids(tokens, tokenizer, model=model, args=args)
    
    if temp_generated_text.endswith('.'):
        temp_generated_text = temp_generated_text.split('.')[-2] + '.'
    else:
        temp_generated_text = temp_generated_text.split('.')[-1]
    
    return temp_generated_text.strip()


def turn_tokens_to_context_clip_text(tokens, tokenizer, model=None, args=None):
    
    temp_generated_text = decode_generated_ids(tokens, tokenizer, model=model, args=args)
    
    all_temp_generated_text = temp_generated_text.split('.')
    if all_temp_generated_text[-1].strip() == '':
        all_temp_generated_text = all_temp_generated_text[:-1]
        
    if len(all_temp_generated_text) == 0:
        return ['']
    
    return all_temp_generated_text



def decode_generated_ids(generated_ids, tokenizer, model=None, args=None):
    model_name = args['model_name']
    if model_name == 'llava_v1_5':
        all_generated_str = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    elif model_name == 'blip2_vicuna_instruct':
        all_generated_str = model.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
    elif model_name == 'mplug_owl2':
        all_generated_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return all_generated_str

def get_dataset(args):
    dataset_name = args['dataset_name']
    data_path = args['data_path']
    if dataset_name == 'mscoco_captions':
        from data.coco import CocoDataset
        # ds = dset.CocoCaptions(root = f'{data_path}/val2014/',
        #                     annFile = f'{data_path}/coco_test_karpathy.json')
        ds = CocoDataset(root = f'{data_path}/val2014/',
                            annFile = f'{data_path}/coco_test_karpathy.json')
    elif dataset_name == 'nocaps':
        from data.nocaps import NoCapsDataset
        domain_name = args['domain']
        ds = NoCapsDataset(dataset_dir=data_path, domain=domain_name)
    elif dataset_name == 'mmvet':
        from data.mmvet import MMVetDataset
        ds = MMVetDataset(data_path=data_path, qa_file=args['mmvet_path'])
    
    return ds
