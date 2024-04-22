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

    seed = args['seed']
    device = args['device']
    test_sample_num = args['test_sample_num']
    
    algo_name = args['algo_name']
        
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
    is_resume = False
    if os.path.exists(saved_file_path): 
        if args['resume']:
            logging.info('output file already exist, resume running!')
            is_resume = True
        elif not args['force_rerun']:
            logging.info('output file already exist, skip!')
            logging.info(saved_file_path)
            logging.info('if you want to rerun, please pass in force_rerun=True')
            # exit()
            return None
        elif args['force_rerun']:
            logging.info('output file already exist, force rerun!')
            is_resume = False
            os.remove(saved_file_path)
            
        
    model, vis_processor, tokenizer = load_model(model_name=model_name, model_type=model_type, device=device)

    
    ds = get_dataset(args)

    
    clip_scorer = CLIPModel(model_name=args['clip_model_name'], model_pretrain=args['clip_model_pretrain'], device=device)

    if args['image_ids_path'] != '' and test_sample_num > 0:
        with open(args['image_ids_path'], 'r') as file:
            loaded_ids = json.load(file)
        if dataset_name == 'mscoco_captions':
            real_indexs = []
            for image_id in loaded_ids:
                if int(image_id) in ds.ids:
                    real_indexs.append(ds.ids.index(int(image_id)))
                
   
            if len(real_indexs) < test_sample_num:
                logging.info('not enough samples, only {} samples'.format(len(real_indexs)))
                return None
            # shuffle
            np.random.seed(seed)
            real_indexs = np.random.permutation(real_indexs)
            real_indexs = real_indexs[:test_sample_num]
            assert len(real_indexs) == test_sample_num
            

    
    elif test_sample_num > 0:
        sample_len = test_sample_num
        indexs = np.arange(len(ds))
        np.random.seed(seed)
        real_indexs = np.random.permutation(indexs)[:sample_len]
    else:
        real_indexs = np.arange(0, len(ds))
    
    real_indexs.sort()
    print('total samples: ', len(real_indexs))
    
    if is_resume:
        with open(saved_file_path, 'r') as file:
            loaded_obj = json.load(file)
            loaded_ids = [obj['image_id'] for obj in loaded_obj]
            done_indexs = [ds.get_index_by_image_id(image_id) for image_id in loaded_ids]
            real_indexs = np.setdiff1d(real_indexs, done_indexs)
    
    print('samples to be run: ', len(real_indexs))
    # ouput file
    print('output file: ', saved_file_path)
    logging.info(saved_file_path)
    
    with torch.no_grad():
        corrects = []
        visited_indexs = []
        valid_in_qs = []

        tqdm_index = tqdm(real_indexs)

        for i, real_index in enumerate(tqdm_index):
            image_id = ds.get_image_id(real_index)

            try:
                img, texts = ds[real_index]
               
                q_spec = 'normal'
                
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
                    'index': int(real_index),
                    'image_id': image_id,
                    'question': one_q,
                    'answer': cleaned_ans,
                    'gt_captions': texts,
                }
                saved_file_path = save_paths['output_path']
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
        

    