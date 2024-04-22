import nltk
from nltk import word_tokenize, pos_tag
from tqdm import tqdm
import numpy as np
from datetime import datetime
import csv
import torch
import random
import os

# Current datetime
now = datetime.now()

# Format datetime
formatted_date = now.strftime('%Y-%m-%d-%H:%M')

LLAVA_CONV_MODE = 'plain'
# LLAVA_CONV_MODE = 'llava_v1'

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    


from huggingface_hub import hf_hub_download

def load_model(model_name, model_type, device='cpu', model_base=None):
    
    if model_name == 'blip2':
        from lavis.models import load_model_and_preprocess

        model, vis_processor, tokenizer = load_model_and_preprocess(name="blip2_t5", model_type=model_type, is_eval=True, device=device)
    elif model_name in ['blip2_vicuna_instruct', 'blip2_t5_instruct']:
        from lavis.models import load_model_and_preprocess

        # model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
        model, vis_processor, tokenizer = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
        vis_processor = vis_processor['eval']
        tokenizer = tokenizer['eval']
        tokenizer = model.llm_tokenizer
        
        from model_utils.blip2_vicuna_instruct import rewrite_blip2_generate_func
        
        model = rewrite_blip2_generate_func(model)
        
        
    elif model_name == 'openflamingo':
        from open_flamingo import create_model_and_transforms

        weight_path_map = {
            'mpt-1b-redpajama-200b': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b',
            'mpt-7b': 'openflamingo/OpenFlamingo-9B-vitl-mpt7b'
        }
        
        cross_attn_setting_map = {
            'mpt-1b-redpajama-200b': 1,
            'mpt-7b': 4
        }
        
        model, vis_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=f"anas-awadalla/{model_type}",
            tokenizer_path=f"anas-awadalla/{model_type}",
            cross_attn_every_n_layers=cross_attn_setting_map[model_type],
            # cache_dir=cache_dir  # Defaults to ~/.cache
        )

        checkpoint_path = hf_hub_download(weight_path_map[model_type], "checkpoint.pt")

        model.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif 'llava_v1_5' in model_name:
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from llava.model.builder import load_pretrained_model

        # model_path = 'liuhaotian/llava-v1.5-7b'
        
        model_type_paths = {
            '7b': 'liuhaotian/llava-v1.5-7b',
            '13b': 'liuhaotian/llava-v1.5-13b',
        }
        
        model_path = model_type_paths[model_type]
        
        # model_base = model_base
        # model_base is mm_projectors\
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, vis_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device=device)

    elif 'mplug_owl2' in model_name:
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.model.builder import load_pretrained_model

        model_type_paths = {
            'llama2-7b': 'MAGAer13/mplug-owl2-llama2-7b'
        }
        
        model_path = model_type_paths[model_type]
        
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, vis_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device=device)


    return model, vis_processor, tokenizer

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return "{:.4f} ({:d})".format(self.avg, self.count)
        
class MetricLogger(object):
    def __init__(self, delimiter="  "):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    # def __getattr__(self, attr):
    #     if attr in self.meters:
    #         return self.meters[attr]
    #     return super().__getattr__(attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)



def model_input_tpl_format(question, model_name, spec='normal', content='QA'):
    if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
        
        if content == 'QA':
            template = 'Question: {}\nAnswer: '
        elif content == 'caption':
            # template = '{}'
            template = '{}\n'
        elif 'mmvet' in content:
            template = '{}'
            
        return template.format(question)

    elif model_name == 'llava_v1_5':
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        conv_mode = LLAVA_CONV_MODE
        
        if content == 'QA':
            templates = {
                # 'plain': 'Question:{} Short Answer:',
                'plain': 'Question:{}\nAnswer:',
                'llava_v1': 'Question:{} Give a short answer.',
            }
        elif content == 'caption':
            templates = {
                'plain': '{}',
                'llava_v1': '{}',
            }
        elif 'mmvet' in content:
            templates = {
                'plain': '{}',
                'llava_v1': '{}',
            }

        if spec == 'wo_image':
            template = templates[conv_mode]
        else:
            template =  DEFAULT_IMAGE_TOKEN + '\n' + templates[conv_mode]
            # template =  DEFAULT_IMAGE_TOKEN + templates[conv_mode]

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], template.format(question))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        return prompt
    elif model_name == 'mplug_owl2':
        if content == 'QA':
            template = '<|image|> Question: {}\nAnswer: '
        elif content == 'caption':
            template = '<|image|>{}\n'
        elif 'mmvet' in content:
            template = '<|image|>{}\n'
            
        return template.format(question)


def model_generate_anw(model, model_name, image, question, image_processor, tokenizer, device='cuda', spec='normal'):
    if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
        
        image = image_processor(image.convert('RGB')).unsqueeze(0).to(device)
        
        if spec == 'wo_image' or spec == 'w_constimg[0]':
            image = torch.zeros_like(image).to(device)
        if spec == 'w_constimg[1]':
            image = torch.ones_like(image).to(device)
        
        # greedy fix
        tq_answers = model.generate({
            'image': image,
            'prompt': question,
        }, num_beams=1, repetition_penalty=1)
        
        return tq_answers[0]
    
    elif model_name == 'llava_v1_5':
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        # conv_mode = 'plain'
        conv_mode = LLAVA_CONV_MODE
        
        
        conv = conv_templates[conv_mode].copy()
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if spec == 'w_constimg[0]':
            image_tensor = torch.zeros_like(image_tensor)
        elif spec == 'w_constimg[1]':
            image_tensor = torch.ones_like(image_tensor)
        
        config = {
            'temperature': 0.2,
            'top_p': None,
            'num_beams': 1,
        }
        
        with torch.inference_mode():

            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                # do_sample=True,
                do_sample=False, # set to deterministic
                temperature=config['temperature'],
                top_p=config['top_p'],
                num_beams=config['num_beams'],
                # no_repeat_ngram_size=3,
                # max_new_tokens=1024,
                max_new_tokens=500,
                output_attentions=True,
                use_cache=True)
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs
    elif 'mplug_owl2' in model_name:
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,  # no sampling. deterministic
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_beams=1, repetition_penalty=1.0,
                use_cache=True,
            )
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs



def model_generate_anw_raw(model, model_name, image, question, image_processor, tokenizer, device='cuda', spec='normal'):
    if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
        
        # if image is Image, then convert ot RGB
        # if isinstance(image, Image.Image):
        #     image = image.convert('RGB')

        image = image_processor["eval"](image.convert('RGB')).unsqueeze(0).to(device)
        
        if spec == 'wo_image' or spec == 'w_constimg[0]':
            image = torch.zeros_like(image).to(device)
        if spec == 'w_constimg[1]':
            image = torch.ones_like(image).to(device)

        tq_answers = model.generate({
            'image': image,
            'prompt': question,
        })
        
        return tq_answers[0]
    
    elif model_name == 'openflamingo':
        vision_x = [image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        if spec == 'w_constimg[0]':
            vision_x = torch.zeros_like(vision_x)
        if spec == 'w_constimg[1]':
            vision_x = torch.ones_like(vision_x)

        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
      
        lang_x = tokenizer(
            [question],
            return_tensors="pt",
        )
        generated_text = model.generate(
            vision_x=vision_x.to(device),
            lang_x=lang_x["input_ids"].to(device),
            attention_mask=lang_x["attention_mask"].to(device),
            max_new_tokens=20,
            num_beams=3,
        )
        
        return tokenizer.decode(generated_text[0])

    elif model_name == 'llava_v1_5':
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        # conv_mode = 'plain'
        conv_mode = LLAVA_CONV_MODE
        
        
        conv = conv_templates[conv_mode].copy()
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if spec == 'w_constimg[0]':
            image_tensor = torch.zeros_like(image_tensor)
        elif spec == 'w_constimg[1]':
            image_tensor = torch.ones_like(image_tensor)
        
        config = {
            'temperature': 0.2,
            'top_p': None,
            'num_beams': 1,
        }
        
        with torch.inference_mode():
            
            outputs = model(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs
    
    elif 'mplug_owl2' in model_name:
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        # https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2/README.md
        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,  # no sampling. deterministic
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs

import re
def model_output_clean(model_name, output, final_question):
    if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:
        return output
    elif model_name == 'openflamingo':
        return output.replace(final_question, '').replace('Answer:', '').replace('<|endofchunk|>', '').strip()
    elif model_name == 'llava_v1_5':
        output = output.replace('Answer:', '').replace('A:', '').strip()
        output = re.sub(r'\n+', ' ', output).strip()
        return output
    elif 'mplug_owl2' in model_name:
        output = output.replace('Answer:', '').replace('A:', '').replace('</s>','').strip()
        output = re.sub(r'\n+', ' ', output).strip()
        return output

def is_contain_other_rel(output, target_rel, rels):
    other_rels = list(set(rels) - set([target_rel]))
    
    output = output.lower()
    
    return any([rel in output for rel in other_rels])

def is_correct_ans(output, target_ans, rels, q_type):
    
    if q_type == 'W_multi_choice':
        bingo_count = 0
        bingo_list = []
        
        sorted_rels = sorted(rels, key=len, reverse=True)
        output = output.lower()
        for rel in sorted_rels:
            if ((len(rel.split()) == 1 and rel in output.split()) or \
                (len(rel.split()) > 1 and rel in output)):
                bingo_count += 1
                bingo_list.append(rel)
                output = output.replace(rel, '').strip()
        
        # if len(bingo_list) == 1 and bingo_list[0] == target_ans:
        #     return True
        
        return len(bingo_list) == 1 and bingo_list[0] == target_ans
        
        # is_contain_target = output.lower().find(target_ans.lower()) > -1
        
        # # if still contain othe answer, then it is wrong
        # output = output.replace(target_ans, '').strip()
        
        # return is_contain_target and (not is_contain_other_rel(output, target_ans, rels))
    elif q_type == 'binary':
        return output.lower().find(target_ans.lower()) > -1


def get_attention(model, model_name, image, question, image_processor, tokenizer, device='cuda', spec='normal'):
    if model_name == 'llava_v1_5':
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        conv_mode = 'plain'
        conv = conv_templates[conv_mode].copy()
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if spec == 'w_constimg[0]':
            image_tensor = torch.zeros_like(image_tensor)
        elif spec == 'w_constimg[1]':
            image_tensor = torch.ones_like(image_tensor)
        
        config = {
            'temperature': 0.2,
            'top_p': None,
            'num_beams': 1,
        }
        
        with torch.inference_mode():

            output = model(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                output_attentions=True)
            
        stack_output = torch.stack(output.attentions)
        
        # out: (layers, heads, tokens, tokens)
        return stack_output.detach().cpu().numpy()


from PIL import Image
import numpy as np

def shuffle_image_in_patches(image, patch_size=14):
    
    # resize to 336px
    image = image.resize((224, 224))
    
    # Load the image
    # image = Image.open(image_path)
    image_width, image_height = image.size
    
    # Convert to NumPy array for block shuffling
    image_array = np.array(image)

    # Calculate the number of patches along width and height
    num_patches_x = image_width // patch_size
    num_patches_y = image_height // patch_size

    # Check if the image dimensions are exactly divisible by the patch size
    if image_width % patch_size != 0 or image_height % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by the patch size.")
    
    # Extract patches
    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            x_start = j * patch_size
            y_start = i * patch_size
            # patches.append(image_array[y_start:y_start + patch_size, x_start:x_start + patch_size, :])
            patches.append(image_array[y_start:y_start + patch_size, x_start:x_start + patch_size])

    # Shuffle patches
    np.random.shuffle(patches)

    # Reconstruct the image from the shuffled patches
    shuffled_image_array = np.zeros_like(image_array)
    for i, patch in enumerate(patches):
        y = (i // num_patches_x) * patch_size
        x = (i % num_patches_x) * patch_size
        # shuffled_image_array[y:y + patch_size, x:x + patch_size, :] = patch
        shuffled_image_array[y:y + patch_size, x:x + patch_size] = patch

    # Convert back to PIL image
    shuffled_image = Image.fromarray(shuffled_image_array)
    return shuffled_image
