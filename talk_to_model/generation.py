import numpy as np
import os
import torch

from talk_to_model.loading import DEVICE


@torch.no_grad()
def generate_response(prompt, model, tokenizer, params):
    prompt = prompt[-2046:]
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    raw_resp = model.generate(input_ids=encoded.input_ids,
                              attention_mask=encoded.attention_mask,
                              **params)
    text_resp = tokenizer.decode(raw_resp[0], skip_special_tokens=True)
    return text_resp

