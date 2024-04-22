import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
from lib.utils import *
import argparse
import torch
import os
import pathlib
from lib.data_utils import *
import pickle
from lib.clip_utils import CLIPModel

from gen.clip_guided import generate_w_clip
from gen.rsp_sampling import rsp_sampling


now = datetime.now()

# Format datetime
formatted_date = now.strftime('%Y-%m-%d-%H:%M')
cache_dir = '~/.cache'


def main(args, logging=None):
    model_name = args['model_name']
    model_type = args['model_type']

    dataset_name = args['dataset_name']
    
    q_type = args['q_type']
    q_content = args['q_content']
    
    algo_name = args['algo_name']

    seed = args['seed']
    
    device = args['device']
    test_sample_num = args['test_sample_num']
        
    set_seed(seed)

    
    save_paths = get_save_paths(args)
    save_file_prefix = save_paths['prefix']
    
    if logging is None:
        log_file_path = './caption_w_clip_logs/{}.log'.format(save_file_prefix)
        pathlib.Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        handlers=[logging.FileHandler(log_file_path, mode='w'),
                                logging.StreamHandler()])

    logging.info('args', args)
    

    saved_file_path = save_paths['output_path']
    if os.path.exists(saved_file_path) and not args['force_rerun']:
        logging.info('output file already exist, skip!')
        logging.info(saved_file_path)
        logging.info('if you want to rerun, please pass in force_rerun=True, it will removed the previous output file')
        return
    elif os.path.exists(saved_file_path) and args['force_rerun']:
        logging.info('output file already exist, but force_rerun=True, so remove and rerun!')
        # remove the file
        os.remove(saved_file_path)
        
    
    model, vis_processor, tokenizer = load_model(model_name=model_name, model_type=model_type, device=device)

    ds = get_dataset(args)
    

    clip_scorer = CLIPModel(model_name=args['clip_model_name'], model_pretrain=args['clip_model_pretrain'], device=device)
    
    if args['task'] == 'eval_mmvet':
        question_set = load_mmvet_question(args['mmvet_path'], args=args)
        print('mmvet question_set loaded: ', len(question_set), flush=True)
        # change q_type and q_content temporarily
        q_type = 'vqa'
        q_content = 'mmvet'
    
    if test_sample_num > 0:
        sample_len = test_sample_num
        image_ids = set([item['image_id'] for item in question_set])
        image_ids = list(image_ids)
        indexs = np.arange(len(image_ids))
        np.random.seed(seed)
        real_indexs = np.random.permutation(indexs)[:sample_len]
        real_indexs = np.sort(real_indexs)
        sub_image_ids = np.array(image_ids)[real_indexs]
        
        sub_question_set = []
        for q_obj in question_set:
            if q_obj['image_id'] in sub_image_ids.tolist():
                sub_question_set.append(q_obj)
        question_set = sub_question_set
        
        
    else:
        real_indexs = np.arange(0, len(ds))
    
    print('question set loaded: ', len(question_set))
    print('output file: ', saved_file_path)
    logging.info(saved_file_path)
    
    with torch.no_grad():
        
        # iterate over all questions
        for q_obj in tqdm(question_set):
            image_id = q_obj['image_id']
            question_id = q_obj['question_id']


            try:
                if dataset_name == 'mmvet':
                    img, texts = ds.get_sample_by_id(question_id)                
                else:
                    img, texts = ds.get_sample_by_id(image_id)
                
                q_spec = 'normal'
                
                if q_type == 'vqa':
                    one_q = q_obj['question']
                else:
                    one_q = create_question(dataset_name, q_type=q_type, q_content=q_content, q_obj={
                        'q_captions': texts,
                    })

                
                final_q = model_input_tpl_format(one_q, model_name, spec=q_spec, content=q_content)

                if algo_name == 'rsp_sampling':
                    gen_func = rsp_sampling
                else:
                    gen_func = generate_w_clip
                
                all_generated_str = gen_func(model, tokenizer, vis_processor, final_q, img, device=device, verbose=False, final_res_num=1, args=args,
                                                                        return_max_probs=True, return_clip_scores=False, clip_scorer=clip_scorer)

                raw_ans = all_generated_str

                cleaned_ans = model_output_clean(model_name, raw_ans[0], final_q)
                
                save_obj = {
                    'question_index': question_id,
                    'image_id': image_id,
                    'question': one_q,
                    'answer': cleaned_ans,
                    'label': q_obj['label'],
                }

                if os.path.exists(saved_file_path):
                    with open(saved_file_path, 'r') as file:
                        loaded_obj = json.load(file)
                        loaded_obj.append(save_obj)
                        save_obj = loaded_obj
                else:
                    save_obj = [save_obj]
                
                with open(saved_file_path, 'w') as file:
                    json.dump(save_obj, file, indent=4)
            except Exception as e:
                logging.info('image_id: ', image_id)
                logging.info('error: ', e)
                continue

    # evaluation
    res = {}
    
    res['config'] = args
    if args['task'] == 'eval_mmvet':
        from data.mmvet import process_output
        res = process_output(saved_file_path)
        save_res_to_json(res, save_paths['res_intermediate_path'])
    

    