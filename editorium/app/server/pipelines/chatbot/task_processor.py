from typing import List

import os
import torch
import random
import json
from tqdm import tqdm
import transformers

from pipelines.chatbot.managed_model import chatbot_models
from task_helpers.progress_bar import ProgressBar



def use_cache(repo_id: str, system_prompt: str, prompt: str, response: str) -> str:
    from hashlib import sha1
    buffer = f'{repo_id}{system_prompt}{prompt}'.encode('utf-8')
    sha1_value = sha1(buffer).hexdigest()
    cache_dir = '/app/output_dir/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'chatbot_cache.json'
    cache_path = os.path.join(cache_dir, cache_file)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        if sha1_value in cache:
            return cache[sha1_value]
    if len(cache.keys()) > 20:
        cache = {}

    if response:
        cache[sha1_value] = response
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    return response
        

def generate_text(repo_id: str, 
                  model_name: str, 
                  template: str,
                  context: str, 
                  prompt: str,
                  max_new_tokens: int = 512,
                  temperature: float = 1,
                  top_p: float = 1,
                  top_k: int = 0,
                  repetition_penalty: float = 1,
                  response_after: str = '',
                  globals: dict = {}) -> dict:
    response = use_cache(repo_id, context, prompt, '')
    if response:
        ProgressBar.set_title('[chatbot] - Using cached response')
        return {
            "texts": [response],
        }

    chatbot_models.load_models(repo_id=repo_id, model_name=model_name)
    template = template.replace('\\n', '\n')
    if not '{context}' in template or not '{input}' in template:
        raise ValueError("Template must contain {context} and {input} placeholders.")
        
    prompt = template.format(context=context, input=prompt)
    
    ProgressBar.set_title('[chatbot] - generating the text')

    class CanceledChecker:
        def __call__(self, *args, **kwargs) -> bool:
            try:
                ProgressBar.noop()
            except Exception:
                return True
            return False # TODO return True if the task should be canceled

    
    inputs = chatbot_models.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=True).to('cuda:0')
    output = chatbot_models.model.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stopping_criteria=transformers.StoppingCriteriaList() + [
                CanceledChecker()
            ]
        )[0]

    new_tokens = len(output) - len(inputs[0])
    response = chatbot_models.tokenizer.decode(output[-new_tokens:], True)

    if 'ASSISTANT:' in response_after:
        response = response.split('\n');
        for i, v in enumerate(response):
            if not v.startswith(response_after):
                continue
            v = v.split(response_after, maxsplit=1)
            response[i] = v[0] if len(v) < 2 else v[1]
        response = '\n'.join(response)
    
    use_cache(repo_id, context, prompt, response)

    return {
        "texts": [response],
    }
    

def process_workflow_task(input: dict, config: dict) -> dict:
    return generate_text(
        repo_id=config['repo_id'],
        model_name=config['model_name'],
        template=config['template'],
        context=config['context'],
        prompt=config['prompt'],
        max_new_tokens=config.get('max_new_tokens', 512),
        temperature=config.get('temperature', 1),
        top_p=config.get('top_p', 1),
        top_k=config.get('top_k', 0),
        repetition_penalty=config.get('repetition_penalty', 1),
        response_after=config.get('response_after', ''),
        globals=config.get('globals', {})
    )
    