from typing import List

import os
import torch
import random
import json
from tqdm import tqdm
import transformers

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.common.task_result import TaskResult
from pipelines.chatbot.managed_model import chatbot_models


class CanceledChecker:
    def __call__(self, *args, **kwargs) -> bool:
        return False # TODO return True if the task should be canceled

def use_cache(repo_id: str, system_prompt: str, prompt: str, response: str) -> str:
    from hashlib import sha1
    buffer = f'{repo_id}{system_prompt}{prompt}'.encode('utf-8')
    sha1_value = sha1(buffer).hexdigest()
    cache_dir = '/app/output_dir/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'chatbot_cache.json'
    if os.path.exists(os.path.join(cache_dir, cache_file)):
        with open(os.path.join(cache_dir, cache_file), 'r') as f:
            cache = json.load(f)
        if sha1_value in cache:
            return cache[sha1_value]
    if len(cache.keys()) > 20:
        cache = {}
    if response:
        cache[sha1_value] = response
        with open(os.path.join(cache_dir, cache_file), 'w') as f:
            json.dump(cache, f, indent=2)
    return response
        


def generate_text(base_dir: str,
                  name: str,
                  repo_id: str, 
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
                  callback: callable = None,
                  globals: dict = {}) -> dict:
    debug_enabled = globals.get('debug', False)
    response = use_cache(repo_id, context, prompt, '')
    if response:
        if debug_enabled:
            text_path = os.path.join(base_dir, f'{name}.txt')
            with open(text_path, 'w') as f:
                f.write(response)
        else:
            text_path = ''
        return {
            "default": response,
            "filepath": text_path,
        }

    chatbot_models.load_models(repo_id=repo_id, model_name=model_name)
    template = template.replace('\\n', '\n')
    if not '{context}' in template or not '{input}' in template:
        raise ValueError("Template must contain {context} and {input} placeholders.")
        
    prompt = template.format(context=context, input=prompt)
    
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
    
    debug_enabled = globals.get('debug', False)
    if debug_enabled:
        text_path = os.path.join(base_dir, f'{name}.txt')
        with open(text_path, 'w') as f:
            f.write(response)
    else:
        text_path = ''

    return {
        "default": response,
        "filepath": text_path,
    }
    

def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False
    
    return generate_text(
        base_dir=base_dir,
        name=name,
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
        callback=callback,
        globals=config.get('globals', {})
    )
    