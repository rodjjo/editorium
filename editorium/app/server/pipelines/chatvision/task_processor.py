from typing import List, Dict

import os
import json
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
    prompt.append('next-turn:')
    context_dict = {}
    prompt_texts = []
    in_turn = True
    in_context = False
    turn_text = ''
    context_text = ''
    context_name = ''
    for line in prompt:
        line = line.strip()
        if line.lower().startswith('context:'):
            if in_turn:
                if turn_text:
                    prompt_texts.append(turn_text)
                    turn_text = ''
            in_context = True
            in_turn = False
            if context_name:
                context_dict[context_name] = context_text
                context_text = ''
                context_name = ''
            context_name = line.lower().split(':')[1].strip()
            continue

        if line.lower() == 'next-turn:':
            if in_turn:
                if turn_text:
                    prompt_texts.append(turn_text)
                    turn_text = ''
            in_turn = True
            in_context = False
            if context_name:
                context_dict[context_name] = context_text
                context_text = ''
                context_name = ''

            continue

        if in_context:
            context_text += '\n' + line
            continue
        if in_turn:
            turn_text += '\n' + line
   
    message_texts = []    
    for p in prompt_texts:
        role = 'user'
        user_text = ''
        assistant_text = ''
        messages = []
        continue_message = []
        
        for line in p.split('\n'):
            line = line.strip()
            if line.lower().startswith('to_context:'):
                context_name = line.lower().split(':')[1].strip()
                continue_message.append([
                    {'role': 'to_context', 'content': context_name},
                ])
                continue 
            if line.lower() ==  'user:':
                if assistant_text:
                    messages.append({'role': role, 'content': assistant_text})
                if user_text:
                    messages.append({'role': role, 'content': user_text})
                assistant_text = ''
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
            elif role == 'user':
                user_text += '\n' + line

        if user_text:
            messages.append({'role': role, 'content': user_text})
        if assistant_text:
            messages.append({'role': role, 'content': assistant_text})
        message_texts.append(messages)
        
        if continue_message:
            message_texts += continue_message
            continue_message = []
    chatvision_model.load_models(repo_id=repo_id)
    responses = []
    
    ProgressBar.set_title('[chatvision] - describing the image')
    with torch.no_grad():
        msgs = []
        for msg_txt in message_texts:
            msgs.append([m for m in msg_txt if m['content'].strip()])
        message_texts = msgs
        
        for i, src_img in enumerate(image):
            ProgressBar.set_progress(i * 100.0/ len(image))
            turn_result = []
            last_response = ''
            for msgs in message_texts:
                should_skip = False
                ignore_image = False
                img = src_img
                for msg in msgs:
                    if '<noimg>' in msg['content']:
                        img = None
                        ignore_image = True
                        msg['content'] = msg['content'].replace('<noimg>', '')
                    if '<skip>' in msg['content']:
                        should_skip = True
                        break
                        
                    while '<from_context:' in msg['content']:
                        start = msg['content'].index('<from_context:')
                        end = msg['content'].index('>', start)
                        if end == -1:
                            print('Error: missing ">" ')
                            break
                        context_name = msg['content'][start + 14:end].lower()
                        if context_name in context_dict:
                            msg['content'] = msg['content'][:start] + context_dict[context_name] + msg['content'][end + 1:]
                        else:
                            msg['content'] = msg['content'][:start] + msg['content'][end + 1:]                    

                if should_skip:
                    continue
                
                continue_role = ''
                if msgs[0]['role'] == 'to_context':
                    if last_response:
                        context_name =  msgs[0]['content']
                        context_dict[context_name] = last_response
                    continue
                response = chatvision_model.model.chat(
                    image=img,
                    msgs=msgs,
                    #repetition_penalty=1.5,
                    #max_new_tokens=256,
                    tokenizer=chatvision_model.tokenizer,
                    sampling=True, # if sampling=False, beam_search will be used by default
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                last_response = response
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
    