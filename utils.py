import json
import time

from openai import OpenAI
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import torch

def run_llm_llama(messages, temperature, max_tokens, tokenizer, model):
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to('cuda')
   
    attention_mask = torch.ones_like(model_inputs)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        model_inputs,  
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,

    )
    response = outputs[0][model_inputs.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result


def run_llm_gpt(messages, temperature, max_tokens, engine):
    
    client = OpenAI()
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
            
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"OpenAI API call failed: {str(e)}")
                return ""
            print(f"OpenAI API error, retrying ({retry_count}/{max_retries})")
            time.sleep(2)


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    else:
        print("dataset not found, you should pick from {cwq, webqsp}.")
        exit(-1)
    return datas, question_string