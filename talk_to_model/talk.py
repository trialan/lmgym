from talk_to_model.generation import generate_response
from talk_to_model.loading import load_causal_lm, load_causal_lm_tokenizer

DEP_PARAMS =     {"max_new_tokens": 64,
                  "do_sample": True,
                  "temperature": 0.72,
                  "repetition_penalty": 1.13125,
                  "top_p": 0.725,
                  "top_k": 0,
                  "eos_token_id": 198,
                  "pad_token_id": 50256}
N_MESSAGES = 10
MODEL_PATH = "Jellywibble/retry-continue-xl-gptj-6500-1002"
TOKENIZER_PATH = "ChaiML/litv2-6B-rev2"

if __name__ == '__main__': 
    model = load_causal_lm(MODEL_PATH)
    tokenizer = load_causal_lm_tokenizer(TOKENIZER_PATH)
    params = DEP_PARAMS
    base_prompt = "This is a friendly conversation between User and Bot. \nUser: How are you today?\nBot: Great thank you!\nUser: How is the family?\nBot: They are doing fantastic!\n"

    convo_history = []
    while len(convo_history) < 2*N_MESSAGES:
        user_message = "User: "+ input("User: ")
        convo_history.append(user_message)
        prompt = base_prompt + "\n".join(convo_history) + "\nBot:"
        completion = generate_response(prompt, model, tokenizer, params)
        bot_message = "Bot:"+completion.split(":")[-1][:-1]
        convo_history.append(bot_message)
        print(bot_message)


