import requests
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

DEVICE = 0
TRUNCATION_SIDE = "left"
RM_PADDING_SIDE = "right"
CAUSAL_PADDING_SIDE = "left"
PAD_TOKEN_ID = 50256


print(f"""Truncation side: {TRUNCATION_SIDE}
RM padding side: {RM_PADDING_SIDE}
CausalLM padding side: {CAUSAL_PADDING_SIDE}""")


def load_causal_lm(hf_path):
    causal_lm = AutoModelForCausalLM.from_pretrained(
            hf_path,
            use_auth_token=True)
    causal_lm.pad_token_id = PAD_TOKEN_ID
    causal_lm = causal_lm.to(DEVICE)
    return causal_lm


def load_evaluation_data(n_samples):
    dataset = load_dataset("ChaiML/user_model_inputs_full_context",
                           split="train",
                           use_auth_token=True)
    evaluation_inputs = dataset['reward_full_input'][:n_samples]
    return evaluation_inputs


def load_reward_model(rm_path):
    reward_model = AutoModelForSequenceClassification.from_pretrained(
            rm_path,
            use_auth_token=True)
    reward_model = reward_model.to(DEVICE)
    return reward_model


def load_reward_tokenizer(rm_tokenizer_path):
    reward_tokenizer = AutoTokenizer.from_pretrained(
            rm_tokenizer_path,
            use_auth_token=True)
    reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
    reward_tokenizer.truncation_side = TRUNCATION_SIDE
    reward_tokenizer.padding_side = RM_PADDING_SIDE
    return reward_tokenizer


def load_causal_lm_tokenizer(tokenizer_path):
    reward_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_auth_token=True)
    reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
    reward_tokenizer.truncation_side = TRUNCATION_SIDE
    reward_tokenizer.padding_side = CAUSAL_PADDING_SIDE
    return reward_tokenizer


