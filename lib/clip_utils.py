import open_clip
import torch
import torch.nn.functional as F

class CLIPModel:
    def __init__(self, model_name='ViT-SO400M-14-SigLIP-384', model_pretrain='webli', device='cuda'):
        
  
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrain, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model.to(device)
        self.model.eval()
        self.device = device
    
    def split_texts(self, texts):
        # texts: list of string
        outputs = []
        for text in texts:
            text = text.replace('\n', '').replace('\t', '').replace('\r', '')
            # split text by '.', '?', '!'
            text = text.replace('?', '.').replace('!', '.')
            text = text.split('.')
            outputs.append(text)
        
        return outputs
        
    @torch.no_grad()
    def get_clip_score(self, text, image):
        clip_vis_enc_model = self.model
        device = self.device
        clip_vis_enc_preprocess = self.preprocess
        clip_vis_enc_tokenizer = self.tokenizer
        
        clip_vis_enc_model.eval()
        clip_vis_enc_model.to(device)
        
        with torch.no_grad():
            text_tokens = clip_vis_enc_tokenizer([text])
            image_input = clip_vis_enc_preprocess(image).to(device)
            image_features = clip_vis_enc_model.encode_image(image_input.unsqueeze(0).to(device))
            text_features = clip_vis_enc_model.encode_text(text_tokens.to(device))
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            scores =  text_features @ image_features.T
            

        return scores.squeeze().item()

    @torch.no_grad()
    def get_clip_score_batch(self, texts, image):
        
        if isinstance(texts, str):
            texts = [texts]
        
        clip_vis_enc_model = self.model
        device = self.device
        clip_vis_enc_preprocess = self.preprocess
        clip_vis_enc_tokenizer = self.tokenizer
        
        clip_vis_enc_model.eval()
        clip_vis_enc_model.to(device)
        
        with torch.no_grad():
            text_tokens = clip_vis_enc_tokenizer(texts)
            image_input = clip_vis_enc_preprocess(image).to(device)
            image_features = clip_vis_enc_model.encode_image(image_input.unsqueeze(0).to(device))
            text_features = clip_vis_enc_model.encode_text(text_tokens.to(device))
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            scores =  text_features @ image_features.T
                    
        return scores.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def get_context_clip_score(self, texts, image):
        clip_vis_enc_model = self.model
        device = self.device
        clip_vis_enc_preprocess = self.preprocess
        clip_vis_enc_tokenizer = self.tokenizer
        
        clip_vis_enc_model.eval()
        clip_vis_enc_model.to(device)
        
        with torch.no_grad():
            text_tokens = clip_vis_enc_tokenizer(texts)
            image_input = clip_vis_enc_preprocess(image).to(device)
            image_features = clip_vis_enc_model.encode_image(image_input.unsqueeze(0).to(device))
            text_features = clip_vis_enc_model.encode_text(text_tokens.to(device))
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            scores =  text_features @ image_features.T
            
        
        mean_scores = scores.cpu().numpy().mean()
        
        return mean_scores.item()
    def get_long_context_clip_score(self, text, image):
        texts = self.split_texts([text])
        
        return self.get_context_clip_score(texts[0], image)
    
    
@torch.no_grad()
def get_clip_score(word, image):
    clip_model = CLIPModel()
    return clip_model.get_clip_score(word, image)