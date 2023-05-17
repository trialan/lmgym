import numpy as np
import os
import torch

from talk_to_model.loading import DEVICE


def generate_davinci_response(prompt):
    prompt = prompt[-2046:]
    out = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=64,
            temperature=1.0,
            stop=["\n", "<|endoftext|>"],
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
    return out['choices'][0]['text']


@torch.no_grad()
def generate_response(prompt, model, tokenizer, params):
    prompt = prompt[-2046:]
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    raw_resp = model.generate(input_ids=encoded.input_ids,
                              attention_mask=encoded.attention_mask,
                              **params)
    text_resp = tokenizer.decode(raw_resp[0])
    return text_resp


def format_string_for_eval_model_input(string):
    lines = [l for l in string.split('\n') if len(l) > 0]
    if len(lines) < 4:
        return None

    res = []

    bot = 1
    for line in lines[::-1]:
        user = 'User' if bot == 0 else 'Bot'
        if ':' not in line:
            break
        pos = line.find(':')
        newline = '{}:{}'.format(user, line[pos + 1:])
        res.insert(0, newline)
        bot ^= 1

    return '\n'.join(res)
