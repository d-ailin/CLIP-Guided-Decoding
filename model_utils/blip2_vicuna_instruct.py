import torch

def prepare_input(model, text, image):

    model.llm_tokenizer.padding_side = "left"

    prompt = text

    bs = image.size(0)

    if isinstance(prompt, str):
        prompt = [prompt] * bs
    else:
        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."


    query_tokens = model.query_tokens.expand(bs, -1, -1)
    if model.qformer_text_input:

        text_Qformer = model.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

    # For video data
    if image.dim() == 5:
        inputs_llm, atts_llm = [], []
        for j in range(image.size(2)):
            this_frame = image[:,:,j,:,:]
            with model.maybe_autocast():
                frame_embeds = model.ln_vision(model.visual_encoder(this_frame))
            frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if model.qformer_text_input:
                frame_query_output = model.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            else:
                frame_query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            frame_inputs_llm = model.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
            frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
            inputs_llm.append(frame_inputs_llm)
            atts_llm.append(frame_atts_llm)
        inputs_llm = torch.cat(inputs_llm, dim=1)
        atts_llm = torch.cat(atts_llm, dim=1)
    else:
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if model.qformer_text_input:
            query_output = model.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = model.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

    llm_tokens = model.llm_tokenizer(
        prompt,
        padding="longest",
        return_tensors="pt"
    ).to(image.device)

    with model.maybe_autocast():
        text_embeds = model.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, text_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
        img_atts = atts_llm
        text_atts = llm_tokens.attention_mask
    
    input_obj = {
        'input_ids': llm_tokens.input_ids,
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
        'img_atts': img_atts,
        'text_atts': text_atts,
        'image_embeds': inputs_llm,
        'text_embeds': text_embeds,
    }

    return input_obj


@torch.no_grad()
def blip_generate(model, text=None, image=None, input_ids=None, input_obj=None, **kwargs):
    '''
        main entry
    '''
    
    if input_obj is None:
        input_obj = prepare_input(model, text, image)
    
    if input_ids is None:
        input_ids = input_obj['input_ids']
    img_atts = input_obj['img_atts']
    text_atts = torch.ones(input_ids.size(), dtype=torch.long).to(model.device)
    
    image_embeds = input_obj['image_embeds']
    

    image_embeds = image_embeds
    text_embeds = model.llm_model.get_input_embeddings()(input_ids).to(model.device)
    inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1).to(model.device)
    attention_mask = torch.cat([img_atts, text_atts], dim=1)
    

    
    with model.maybe_autocast():

        
        outputs = model.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )
    
    # if output is a tensor, then it is the generated ids
    # if output is a dict, then it contains more information
    if isinstance(outputs, torch.Tensor):
        outputs[outputs == 0] = 2
    
    return outputs

    

'''
    https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_vicuna_instruct.py
    rewrite blip2_generate_func to accept more parameters
'''
@torch.no_grad()
def generate_func(
        model,
        samples,
        **kwargs,
    ):
    self = model
    
    self.llm_tokenizer.padding_side = "left"

    if "prompt" in samples.keys():
        prompt = samples["prompt"]
    else:
        prompt = self.prompt

    image = samples["image"]

    bs = image.size(0)

    if isinstance(prompt, str):
        prompt = [prompt] * bs
    else:
        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

    # For TextCaps
    if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
        prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

    query_tokens = self.query_tokens.expand(bs, -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

    # For video data
    if image.dim() == 5:
        inputs_llm, atts_llm = [], []
        for j in range(image.size(2)):
            this_frame = image[:,:,j,:,:]
            with self.maybe_autocast():
                frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
            frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                frame_query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            else:
                frame_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
            frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
            inputs_llm.append(frame_inputs_llm)
            atts_llm.append(frame_atts_llm)
        inputs_llm = torch.cat(inputs_llm, dim=1)
        atts_llm = torch.cat(atts_llm, dim=1)
    else:
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

    llm_tokens = self.llm_tokenizer(
        prompt,
        padding="longest",
        return_tensors="pt"
    ).to(image.device)

    with self.maybe_autocast():
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )
    
    outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
    

    return outputs
    # return output_text

def rewrite_blip2_generate_func(model):
    model.new_generate = generate_func

    return model
