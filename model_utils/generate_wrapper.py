import torch
from model_utils.blip2_vicuna_instruct import blip_generate

class Wrapper():
    def __init__(self, model, args):
        self.model_name = args['model_name']
        self.model = model
        self.cache = {}
        
    def add_cache(self, obj):
        self.cache.update(obj)
    
    def generate(self, input_ids=None, images=None, return_prob=False, **kwargs):
        model_name = self.model_name
        model = self.model
        
        if model_name in ['llava_v1_5', 'mplug_owl2']:
            
            if return_prob:
                up_params = {
                    'output_scores': True,
                    'return_dict_in_generate': True,
                }
            kwargs.update(up_params)
            
            outputs = model.generate(
                input_ids,
                images=images,
                use_cache=True,
                **kwargs
            )

            final_output = self.process_output(outputs, start_index=input_ids.shape[1])
            
            return final_output
        elif model_name in ['blip2_vicuna_instruct']:
            if return_prob:
                up_params = {
                    'output_scores': True,
                    'return_dict_in_generate': True,
                }
            kwargs.update(up_params)
            
            
            input_obj = self.cache['input_obj']
            outputs = blip_generate(model, input_ids=input_ids, input_obj=input_obj, **kwargs,
                                    pad_token_id=32000) # important to set pad_token_id!
            
            inter_output = self.process_output(outputs, start_index=0) # they directly return the trimmed output_ids

            batch_input_ids = input_ids.expand(inter_output['output_ids'].shape[0], -1)
            inter_output['output_ids'] = torch.cat([batch_input_ids, inter_output['output_ids']], 1)

            final_output = inter_output
            return final_output
        
    def process_output(self, outputs, start_index=0):
        '''
        outputs: dict
            {
                'sequences': list of tensor, (num_samples, seq_len)
                'scores': tuple of tensor, (seq_len, num_samples, vocab_size)
            }
        '''
        final_output = {}
        
        
        if (not isinstance(outputs, torch.Tensor)) and 'scores' in outputs.keys():
            scores = outputs['scores']
            logits = outputs['logits']
            _scores = []
            for i in range(len(scores)): # seq_len, (num_samples, vocab_size)
                _scores.append( torch.softmax(logits[i], -1).unsqueeze(0))
            scores = torch.cat(_scores, 0)
            sequences = outputs['sequences']
            
            # get the token probabilities
            if self.model_name in ['blip2_vicuna_instruct']:
                token_indexs = sequences[:, start_index+1:].t() # they have a 0 at the beginning
            else:
                token_indexs = sequences[:, start_index:].t()
            i1, i2 = torch.meshgrid(torch.arange(scores.shape[0]), torch.arange(scores.shape[1]), indexing='ij')
                        
            output_token_probs = scores[i1, i2, token_indexs].t() # out: (num_samples, num_tokens)
            final_output = {
                'output_ids': sequences,
                'output_token_probs': output_token_probs.detach().cpu().numpy(),
                'trim_output_ids': sequences[:, start_index:],
            }
        else:
            final_output = {
                'output_ids': sequences,
                'trim_output_ids': outputs[:, start_index:],
            }
        return final_output