from lib.data_utils import *
from hallucination_eval.utils.chair import main as chair_main
from hallucination_eval.eval_coco_cap import main as eval_coco_cap_main
from hallucination_eval.ana_hallu_single import main as ana_hallu_single_main
import json


def eval_coco(args):
    
    device = args['device']
    
    save_paths = get_save_paths(args)

    output_path = save_paths['output_path']
    
    res = {}
    chair_res = chair_main(output_path, args['data_path'] + '/annotations/')
    overall_chair_scores = chair_res['overall_metrics']
    res['CHAIRs'] = overall_chair_scores['CHAIRs']
    res['CHAIRi'] = overall_chair_scores['CHAIRi']
    
    # save chair_res to json
    save_res_to_json(chair_res, save_paths['res_intermediate_path'])
    
    coverage_res = ana_hallu_single_main(save_paths['res_intermediate_path'], args)
    for k, v in coverage_res.items():
        res[k] = v
        
    
    # based on the chair_res, we can get basic stastic (e.g. avg sentence length) the coverage, and hallucination ratios
    coco_eval_res = eval_coco_cap_main(output_path, device=device, data_path=args['data_path'])
    for k, v in coco_eval_res.items():
        res[k] = v
    
    # save res to json
    final_res = {}
    final_res['res'] = res
    final_res['config'] = args
    
    return final_res

import pandas as pd

from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


from data.nocaps_util import CHAIR as NocapsCHAIR
def eval_nocaps(args):
    '''
        follow https://arxiv.org/pdf/2210.07688.pdf 
        
        only add two
        types of object categories to our final object list: 1)
        super-categories that have sub-categories, and 2)
        object categories that have neither super-category
        nor sub-categories. Eventually, we construct a list
        of 139 coarse-grained object categories from the
        600 classes.
        
        https://github.com/nocaps-org/image-feature-extractors/blob/master/data/oi_categories.json
        https://storage.googleapis.com/openimages/web/download_v7.html#df-classes-hierarchy
        
    '''
    print('evaluating nocaps...')
    save_paths = get_save_paths(args)
    output_path = save_paths['output_path']
    image_ids = []
    for item in json.load(open(output_path, 'r')):
        image_ids.append(item['image_id'])
    
    chair_main = NocapsCHAIR(image_ids, args)
    chair_res = chair_main.compute_chair(output_path)
    
    save_res_to_json(chair_res, save_paths['res_intermediate_path'])
    res = {}
    overall_chair_scores = chair_res['overall_metrics']
    res['CHAIRs'] = overall_chair_scores['CHAIRs']
    res['CHAIRi'] = overall_chair_scores['CHAIRi']
    print('CHAIRs', res['CHAIRs'])
    print('CHAIRi', res['CHAIRi'])
    
    # print('res', res)
    basic_res = ana_hallu_single_main(save_paths['res_intermediate_path'], args)
    for k, v in basic_res.items():
        res[k] = v
    # res['sample_num'] = len(image_ids)
    
    # save res to json
    final_res = {}
    final_res['res'] = res
    final_res['config'] = args
    
    
    return final_res
    

def main(args, logging=None):

    
    dataset_name = args['dataset_name']
    save_paths = get_save_paths(args)
    if dataset_name in ['mscoco_captions', 'mscoco']:
    # if dataset_name == 'mscoco_captions' or dataset_name == 'mscoco':
        final_res = eval_coco(args)
    elif dataset_name == 'nocaps':
        final_res = eval_nocaps(args)
    else:
        print('dataset_name {} not supported yet'.format(dataset_name))
        raise NotImplementedError
    
    save_res_to_json(final_res, save_paths['res_path'])
    print('final res saved to {}'.format(save_paths['res_path']))
    

if __name__ == '__main__':
    args = {
        'dataset_name': 'nocaps',
        'data_path': '/data/ailin/nocaps/',
    }
    
    eval_nocaps(args)
