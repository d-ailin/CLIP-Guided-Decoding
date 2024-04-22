import torch
from lib.data_utils import *
from lib.utils import *
from PIL import Image
import copy

def prepare_sampling_params(args):
    model_name = args['model_name']
    algo_name = args['algo_name']
    sampling_params = args['sampling_params']

    final_sampling_params = copy.deepcopy(sampling_params)
    
    
    return final_sampling_params

def rsp_sampling(model, tokenizer, image_processor, question, image, args=None,  device='cuda', spec='normal', **kwargs):
    
    model_name = args['model_name']
    
    sampling_params = prepare_sampling_params(args)
    
    if model_name in ['blip2', 'blip2_vicuna_instruct', 'blip2_t5_instruct']:

        image = image_processor(image.convert('RGB')).unsqueeze(0).to(device)
        
        output_ids = model.new_generate(model, {
            'image': image,
            'prompt': question,
        }, **sampling_params)
        
        # decoding
        tq_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for i in range(len(tq_answers)):
            tq_answers[i] = tq_answers[i].strip()
        
        return tq_answers

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


        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        
        with torch.inference_mode():

            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                use_cache=True,
                **sampling_params
                )
        
        input_token_len = input_ids.shape[1]
        

        if not isinstance(output_ids, torch.Tensor):
            return output_ids
        
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        
        _outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        _outputs = [output.strip() for output in _outputs]
        outputs = []
        for _output in _outputs:
            if _output.endswith(stop_str):
                _output = _output[:-len(stop_str)]
            _output = _output.strip()
            outputs.append(_output)
        return outputs
        
        # return [outputs]
    
    elif 'mplug_owl2' in model_name:
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)



        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                use_cache=True,
                **sampling_params
            )
        
        if not isinstance(output_ids, torch.Tensor):
            return output_ids

    
        outputs = []
        for output_id in output_ids:
            outputs.append(tokenizer.decode(output_id[input_ids.shape[1]:], skip_special_tokens=True).strip())

        return outputs

if __name__ == '__main__':
    model_name = 'llava_v1_5'
    model_type = '7b'

    # model_name = 'blip2_vicuna_instruct'
    # model_type = 'vicuna7b'
    
    # model_name = 'mplug_owl2'
    # model_type = 'llama2-7b'

    
    device = 'cuda:1'
    print('begin loading model')
    model, vis_processor, tokenizer = load_model(model_name=model_name, model_type=model_type, device=device)
    print('end loading model')

    image_id = '159030'
    # image_id = '230598'
    # image_id = '193021'
    image_id = '565389'
    # image_id = '445668'
    # image_id = '016491'
    image_id = '157001'
    image_id = '134863'
    image_id = '072397'
    image_id = '256192'
    image_path = '/data/ailin/coco/val2014/COCO_val2014_000000{}.jpg'.format(image_id)
    # image_path = '/home/data/coco/val2014/COCO_val2014_000000{}.jpg'.format(image_id)
    
    # image_id = 2
    # image_id = 3
    # image_id = 4
    # image_path = './dalle_imgs/{}.png'.format(image_id)

    image = Image.open(image_path).convert("RGB")

    if model_name == 'llava_v1_5':
        # text = "<image>\nDescribe this image."
        text = "<image>\nDescribe this image in detail."
    elif model_name == 'blip2_vicuna_instruct':
        # text = "Describe this image."
        text = "Describe this image in detail."
    elif model_name == 'mplug_owl2':
        # text = "<|image|>Describe this image.\n"
        text = "<|image|>Describe this image in detail.\n"


    
    args = {
        'model_name': model_name,
        'model_type': model_type,
        'algo_name': 'rsp_sampling',
        'sampling_params': {
            'temperature': 0.2,
            'top_p': 1,
            'top_k': 5,
            'num_beams': 1,
            'do_sample': True,
            'num_return_sequences': 1,
            'max_new_tokens': 500,
        }
    }
    
    # all_generated_str, generated_ids, gen_objs = generate_greedy(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args)
    # all_generated_str, generated_ids, gen_objs = generate_greedy(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)

    # all_generated_str, generated_ids, gen_objs = generate_v0(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)

    # all_generated_str, generated_ids, gen_objs = generate_v1(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)

    # all_generated_str, generated_ids, gen_objs = generate_v2(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)
    # all_generated_str, generated_ids, gen_objs = generate_v3(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)


    # all_generated_str, generated_ids, gen_objs = generate_v4(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)

    all_generated_str = rsp_sampling(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
                                                                 return_max_probs=True, return_clip_scores=True)
    

    # ablation
    # all_generated_str, generated_ids, gen_objs = generate_topk_clip(model, tokenizer, vis_processor, text, image, device=device, verbose=True, final_res_num=1, args=args,
    #                                                              return_max_probs=True, return_clip_scores=True)
    
    # print('all_generated_str', all_generated_str)
    # print('gen_objs', gen_objs)

    for generated_str in zip(all_generated_str):
        # print(generated_str, prob_score, clip_score, max_attn_score, attn_score_w_lasttoken)
        print(generated_str)


