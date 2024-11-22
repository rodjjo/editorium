from typing import List

import os
import torch

from pipelines.chatvision.managed_model import chatvision_model
from task_helpers.progress_bar import ProgressBar
from pipelines.common.utils import ensure_image

class CanceledChecker:
    def __call__(self, *args, **kwargs) -> bool:
        return False # TODO return True if the task should be canceled


def generate_text(repo_id: str, 
                  image,
                  prompt: str,
                  system_prompt: str = '',
                  temperature: float = 0.7,
                  globals: dict = {}) -> dict:
    if not image:
        raise ValueError('Image is required')
    if type(image) is str:
        image = torch.load(image)
   
    image = ensure_image(image)
    
    if type(image) is not list:
        image = [image]
        
    prompt_texts = []
    prompt = prompt.split('\n')
    prompt = [p.strip() for p in prompt if p]
    text = ''
    for p in prompt:
        p = p.strip()
        if p == '':
            continue
        if p.lower() == 'next-turn:':
            prompt_texts.append(text)
            text = ''
        else:
            text += p + '\n'
    if text:
        prompt_texts.append(text)
    
    message_texts = []    
    for p in prompt_texts:
        role = 'user'
        user_text = ''
        assistant_text = ''
        messages = []
        for line in p.split('\n'):
            if line.lower() ==  'user:':
                if assistant_text:
                    messages.append({'role': role, 'content': assistant_text})
                    assistant_text = ''
                if user_text:
                    messages.append({'role': role, 'content': user_text})
                user_text = ''
                role = 'user'
                continue
            if line.lower() == 'assistant:':
                if user_text:
                    messages.append({'role': role, 'content': user_text})
                    user_text = ''
                if assistant_text:
                    messages.append({'role': role, 'content': assistant_text})
                assistant_text = ''
                role = 'assistant'
                continue
            if role == 'assistant':
                assistant_text += '\n' + line
            else:
                user_text += '\n' + line

        if user_text:
            messages.append({'role': role, 'content': user_text})
        if assistant_text:
            messages.append({'role': role, 'content': assistant_text})
        message_texts.append(messages)
      
    chatvision_model.load_models(repo_id=repo_id)
    responses = []
    
    ProgressBar.set_title('[chatvision] - describing the image')
    with torch.no_grad():
        for i, img in enumerate(image):
            ProgressBar.set_progress(i * 100.0/ len(image))
            turn_result = []
            for msgs in message_texts:
                response = chatvision_model.model.chat(
                    image=img,
                    msgs=msgs,
                    #repetition_penalty=1.0,
                    tokenizer=chatvision_model.tokenizer,
                    sampling=True, # if sampling=False, beam_search will be used by default
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                turn_result.append(response)
            for index, response in enumerate(turn_result):
                response = response.split('\n')
                response = [r.strip() for r in response if r.strip()]
                response = '\n'.join([r for r in response if r])
                turn_result[index] = response
                
            responses.append('\n\n'.join(turn_result))

    return {
        "texts": responses,
    }
    

def process_workflow_task(input: dict, config: dict) -> dict:
    image = input.get('default', {}).get('images', [])
    if not image:
        image = input.get('image', {}).get('images', [])
    
    return generate_text(
        repo_id=config['repo_id'],
        image=image,
        prompt=config['prompt'],
        system_prompt=config.get('system_prompt', ''),
        temperature=config.get('temperature', 0.7),
        globals=config.get('globals', {})
    )
    