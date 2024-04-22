import json
import argparse
import numpy as np
# import clip
from PIL import Image
import torch
from lib.clip_utils import CLIPModel
from lib.data_utils import get_dataset
# if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
# else:
#     device = device

# model, preprocess = clip.load("ViT-B/32", device='cuda')
# # model, preprocess = clip.load("ViT-L/14", device=device)

# def compute_clip_score(image_path, caption):
#     # Preprocess the image and prepare the text
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#     text = clip.tokenize([caption]).to(device)

#     # Compute the logits
#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)

#     # Pick the top 5 most similar labels for the image
#     # logits_per_image, _ = model(image, text)
#     # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
#     cos_sim = torch.cosine_similarity(image_features, text_features).detach().cpu().numpy()

#     return cos_sim[0]

def main(json_file_1, args):

    with open(json_file_1, 'r') as f:
        data_1 = json.load(f)['sentences']

        # with open(args.input2, 'r') as f:
        #     data_2 = json.load(f)['sentences']

        # assert len(data_1) == len(data_2)

    generated_gt_word_num = []
    generated_hallucinated_word_num = []
    generated_word_num = []

    dup_generated_gt_word_num = []    
    dup_generated_hallucinated_word_num = []
    dup_generated_word_num = []
    
    coverage_ratios = []
    
    captions = []
    image_ids = []
    
    generated_len = []
    for i in range(len(data_1)):
        
        caption = data_1[i]['caption']
        
        image_ids.append(data_1[i]['image_id'])
        captions.append(caption)
        
        if 'mscoco_gt_words' in data_1[i].keys():
            k_gt_words = 'mscoco_gt_words'
            k_generated_words = 'mscoco_generated_words'
            k_hallucinated_words = 'mscoco_hallucinated_words'
        else:
            k_gt_words = 'valid_gt_words'
            k_generated_words = 'valid_generated_words'
            k_hallucinated_words = 'valid_hallucinated_words'
        
        
        mscoco_gt_words = data_1[i][k_gt_words]
        mscoco_generated_words = data_1[i][k_generated_words]
        mscoco_hallucinated_words = [ item[1] for item in data_1[i][k_hallucinated_words]]
        
        no_duplicate_mscoco_generated_words = np.unique(mscoco_generated_words)
        no_duplicate_mscoco_hallucinated_words = np.unique(mscoco_hallucinated_words)

        duplicate_mscoco_generated_words = mscoco_generated_words
        duplicate_mscoco_hallucinated_words = mscoco_hallucinated_words

        generated_len.append(len(caption.split()))
        generated_gt_word_num.append(len(no_duplicate_mscoco_generated_words) - len(no_duplicate_mscoco_hallucinated_words))
        generated_hallucinated_word_num.append(len(no_duplicate_mscoco_hallucinated_words))
        generated_word_num.append(len(no_duplicate_mscoco_generated_words))
        
        dup_generated_gt_word_num.append(len(duplicate_mscoco_generated_words) - len(duplicate_mscoco_hallucinated_words))
        dup_generated_hallucinated_word_num.append(len(duplicate_mscoco_hallucinated_words))
        dup_generated_word_num.append(len(duplicate_mscoco_generated_words))
        
        correct_generated_words = [w in mscoco_gt_words for w in no_duplicate_mscoco_generated_words]
        if len(mscoco_gt_words) > 0:
            coverage_ratios.append(sum(correct_generated_words)/len(mscoco_gt_words))
        else:
            coverage_ratios.append(0)
        

    print('num of samples:', len(data_1))
    # print('generated_len: {}'.format(np.mean(generated_len)) )
    
    print('no duplicate:')
    print('generated_gt_word_num: {}, ratio: {}'.format(np.mean(generated_gt_word_num), np.mean(generated_gt_word_num)/np.mean(generated_word_num)) )
    print('generated_hallucinated_word_num: {}, ratio: {}'.format(np.mean(generated_hallucinated_word_num), np.mean(generated_hallucinated_word_num)/np.mean(generated_word_num)) )
    print('generated_word_num: {}'.format(np.mean(generated_word_num)) )

    print('duplicate:')
    print('generated_gt_word_num: {}, ratio: {}'.format(np.mean(dup_generated_gt_word_num), np.mean(dup_generated_gt_word_num)/np.mean(dup_generated_word_num)) )
    print('generated_hallucinated_word_num: {}, ratio: {}'.format(np.mean(dup_generated_hallucinated_word_num), np.mean(dup_generated_hallucinated_word_num)/np.mean(dup_generated_word_num)) )
    print('mean coverage ratio: {}'.format(np.mean(coverage_ratios)))
    
    ret = {}
    
    ret['sample_num'] = len(data_1)
    ret['avg_sentence_length'] = np.mean([len(c) for c in captions])
    ret['avg_word_num'] = np.mean([len(c.split()) for c in captions])

    
    ret['avg_generated_gt_word_num'] = np.mean(generated_gt_word_num)
    ret['avg_generated_hallucinated_word_num'] = np.mean(generated_hallucinated_word_num)
    ret['avg_generated_word_num'] = np.mean(generated_word_num)
    
    ret['avg_dup_generated_gt_word_num'] = np.mean(dup_generated_gt_word_num)
    ret['avg_dup_generated_hallucinated_word_num'] = np.mean(dup_generated_hallucinated_word_num)
    ret['avg_dup_generated_word_num'] = np.mean(dup_generated_word_num)
    ret['avg_coverage_ratio'] = np.mean(coverage_ratios)
    
    # compute clip score
    clip_scorer = CLIPModel('ViT-B-32', 'openai', device=args['device'])
    dataset_name = args['dataset_name']
    data_path = args['data_path']
    clip_scores = []
    
    ds = get_dataset(args)
    
    # if dataset_name == 'mscoco_captions':
    #     from data.coco import CocoDataset
    #     # ds = dset.CocoCaptions(root = f'{data_path}/val2014/',
    #     #                     annFile = f'{data_path}/coco_test_karpathy.json')
    #     ds = CocoDataset(root = f'{data_path}/val2014/',
    #                         annFile = f'{data_path}/coco_test_karpathy.json')
    # elif dataset_name == 'nocaps':
    #     from data.nocaps import NoCapsDataset
    #     ds = NoCapsDataset(dataset_dir=data_path, domain='out-domain')
    
    for image_id, caption in zip(image_ids, captions):
        image, _ = ds.get_sample_by_id(image_id)
        clip_score = clip_scorer.get_long_context_clip_score(caption, image)
        
        clip_scores.append(clip_score)
        # ret[image_id] = clip_score
        # print('image_id: {}, clip_score: {}'.format(image_id, clip_score))
    ret['clip_score'] = np.array(clip_scores).mean()
    
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default='data/ana_hallu.json')
    # parser.add_argument('--input2', type=str, default='data/ana_hallu.json')

    args = parser.parse_args()

    json_file_1 = args.input1
    
    main(json_file_1)
    