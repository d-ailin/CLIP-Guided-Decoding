from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import numpy as np
import argparse
import torch
import clip
from PIL import Image
import numpy as np
import traceback
    


def main(res_json, anchor_json=None, device=None, data_path = ''):
    
    ret = {}
    
    if anchor_json is not None:
        anchor_image_ids = []
        with open(anchor_json, 'r') as f:
            anchor_json = json.load(f)
        for item in anchor_json:
            anchor_image_ids.append(item['image_id'])

    with open(res_json, 'r') as f:
        res = json.load(f)
    results = []
    for item in res:
        if anchor_json is not None:
            if item['image_id'] in anchor_image_ids:
                results.append({'image_id': item['image_id'], 'caption': item['answer']})
        else:
            results.append({'image_id': item['image_id'], 'caption': item['answer']})

    print('avg word num:', np.mean([len(item['caption'].split()) for item in results]))
    print('avg sentence length:', np.mean([len(item['caption']) for item in results]))
    
    ret['sample_num'] = len(results)
    ret['avg_sentence_length'] = np.mean([len(item['caption']) for item in results])
    ret['avg_word_num'] = np.mean([len(item['caption'].split()) for item in results])
    
    # # Prepare data for evaluation
    # results = []
    # for img_id in generated_captions:
    #     results.append({'image_id': img_id, 'caption': generated_captions[img_id]})

    # Load and run the MSCOCO evaluator
    # coco = COCO('/home/ailin/temp/Hallucination/annotations/captions_val2014.json')  # Load your actual annotations here
    # coco = COCO('./hallucination_eval/annotations/captions_val2014.json')  # Load your actual annotations here
    caption_path = './hallucination_eval/annotations/captions_val2014.json'
    if data_path != '':
        caption_path = data_path + '/annotations/captions_val2014.json'
    coco = COCO(caption_path)
    coco_res = coco.loadRes(results)
    # print('coco', coco)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()
    
    for metric, score in coco_eval.eval.items():
        print('%s: %.3f'%(metric, score))
        ret[metric] = score

    ### clip score evaluation



    # Load the CLIP model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)

    def compute_clip_score(image_path, caption):
        # Preprocess the image and prepare the text
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([caption]).to(device)

        # Compute the logits
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        # Pick the top 5 most similar labels for the image
        # logits_per_image, _ = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        cos_sim = torch.cosine_similarity(image_features, text_features).detach().cpu().numpy()

        return cos_sim[0]

    clip_scores = []
    for item in results:
        image_path = '/data/ailin/coco/val2014/' + coco.loadImgs(item['image_id'])[0]['file_name']
        
        if data_path != '':
            image_path = data_path + '/val2014/' + coco.loadImgs(item['image_id'])[0]['file_name']
        
        cap = item['caption']
        # split cap by '.', '?', '!'
        cap = cap.replace('?', '.').replace('!', '.').replace('–', '.')
        caps = cap.split('.')
        cap_clip_scores = []
        # print(caps)
        for c in caps:
            try:
                clip_score = compute_clip_score(image_path, c)
                cap_clip_scores.append(clip_score)
            except:
                print('image_id: {} error'.format(item['image_id']))
                traceback.print_exc()
                
                # cap_clip_scores.append(0)
                
        # clip_score = compute_clip_score(image_path, item['caption'])
        # print(_clip_score)
        # cap_clip_scores = 
        clip_scores.append(np.array(cap_clip_scores).mean())
    # Example usage
    # image_path = 'path_to_your_image.jpg'
    # caption = 'Your generated caption'
    # clip_score = compute_clip_score(image_path, caption)
    # print(f"CLIP Score: {clip_score}")
    clip_scores = np.array(clip_scores)
    print('clip score:', np.mean(clip_scores))
    ret['clip_score'] = float(np.mean(clip_scores))


    # compute CHAIR score
    # from hallucination_eval.utils.chair import main as chair_main
    # print('computing CHAIR metric...')

    # annotation_p = './hallucination_eval/annotations/'
    # chair_main(res_json, annotation_p, anchor_file=anchor_json)

    print('returning results...')

    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_json', type=str, default='./caption_w_clip_logs/mscoco_captions_llava_v1_5_7b_describe_caption_fast_test_dosampleF_0_output.json')
    parser.add_argument('--anchor_json', type=str, default=None)

    args = parser.parse_args()

    # load captions from file
    # res_json = './caption_w_clip_logs/mscoco_captions_llava_v1_5_7b_describe_caption_fast_test_dosampleF_0_output.json'
    # res_json = './caption_logs/mscoco_captions_llava_v1_5_7b_describe_caption_fast_test_dosampleF_0_output.json'

    res_json = args.res_json
    anchor_json = args.anchor_json

    main(res_json, anchor_json)