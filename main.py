from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
from caption import main as clip_g_main
from datetime import datetime
from eval_main import main as eval_main
from qa import main as qa_main

def build_config(raw_config):
    config = {}
    cfg = raw_config['run']
    
    config['task'] = raw_config['task']
    config['force_rerun'] = raw_config['force_rerun']
    config['resume'] = raw_config['resume']
    
    config['dataset_name'] = cfg['dataset_name']
    
    if config['dataset_name'] == 'nocaps':
        config['domain'] = cfg['domain']
    
    config['data_path'] = cfg['data_path']
    config['q_type'] = cfg['q_type']
    config['q_content'] = cfg['q_content']


    config['model_name'] = cfg['qa_model']['model_name']
    config['model_type'] = cfg['qa_model']['model_type']
    config['algo_name'] = cfg['algo']['name']
    config['algo_version'] = cfg['algo']['version']
    config['clip_model_name'] = cfg['algo']['clip']['model_name']
    config['clip_model_pretrain'] = cfg['algo']['clip']['model_pretrain']
    
    config['seed'] = cfg['seed']
    config['test_sample_num'] = cfg['test_sample_num']
    
    config['tag'] = cfg.get('tag', '')
    config['device'] = cfg['device']
    
    
    config['image_ids_path'] = cfg.get('image_ids_path', '')
    
    if config['algo_name'] in ['rsp_sampling', 'clip_guided']:
        config['using_sampling_params'] = True
        config['sampling_params'] = cfg.get('algo', {}).get('sampling', {})

    else:
        config['using_sampling_params'] = False
        
    config['eval_seem_labels'] = cfg.get('eval_seem_labels', False)
    
    config['mmvet_path'] = cfg.get('mmvet_path', '')
    
    if config['algo_name'] in ['clip_guided']:
        config['using_scoring_params'] = True
    else:
        config['using_scoring_params'] = False
    if config['using_scoring_params']:
        config['scoring'] = cfg.get('algo', {}).get('scoring', {})
    
    
    
    return config

now = datetime.now()
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

    _raw_config = OmegaConf.to_yaml(cfg, resolve=True)
    print(_raw_config)
    _conf = OmegaConf.create(_raw_config)
    main_conf = build_config(_conf)

    hydra_context = HydraConfig.get()
    
    logger.info('begin exp:')
    print('log saved at {}'.format(hydra_context.runtime.output_dir))
    # log config
    logger.info('config:\n' + OmegaConf.to_yaml(main_conf, resolve=True) )

    if main_conf['task'] == 'generation':
        clip_g_main(main_conf, logger)
    elif main_conf['task'] == 'eval':
        eval_main(main_conf, logger)
    elif main_conf['task'] == 'eval_mmvet':
        qa_main(main_conf, logger)
        
    print('log saved at {}'.format(hydra_context.runtime.output_dir))
    logger.info('end exp.')


if __name__ == "__main__":
    run()