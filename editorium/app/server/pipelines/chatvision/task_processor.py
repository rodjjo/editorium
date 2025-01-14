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
    
    context_dict = {}
    prompt_texts = []
    prompt = prompt.split('\n')
    prompt = [p.strip() for p in prompt if p]
    text = ''
    context_name = ''
    in_context_capture = True
    for p in prompt:
        p = p.strip()
        if p == '':
            continue
        if in_context_capture and p.lower().startswith('context:'):
            if context_name:
                context_dict[context_name] = text
                text = ''
            context_name = p.split(':', 1)[1].strip().lower()
            if not context_name:
                context_name = 'default'
        elif p.lower() == 'next-turn:':
            in_context_capture = False
            if not text:
                context_name = ''
                continue
            if context_name:
                context_dict[context_name] = text
            else:
                prompt_texts.append(text)
            context_name = ''
            text = ''
        else:
            text += p + '\n'
    if text:
        if context_name:
            context_dict[context_name] = text
        else:
            prompt_texts.append(text)
    
    message_texts = []    
    for p in prompt_texts:
        while '<context:' in p:
            start = p.index('<context:')
            end = p.index('>', start)
            if end == -1:
                print('Error: missing ">" ')
                break
            context_name = p[start + 9:end].lower()
            if context_name in context_dict:
                p = p[:start] + context_dict[context_name] + p[end + 1:]
            else:
                p = p[:start] + p[end + 1:]
        role = 'user'
        user_text = ''
        assistant_text = ''
        continue_text = ''
        last_assistant_text = ''
        messages = []
        continue_message = []
        
        def add_continue_message():
            nonlocal continue_text
            if continue_text:
                continue_message.append([
                    {'role': role, 'content': '.'},
                    {'role': 'user', 'content': continue_text},
                ])
                continue_text = ''

        for line in p.split('\n'):
            if line.lower() == 'continue:' or line.lower() == 'continue-chain:':
                if assistant_text:
                    messages.append({'role': role, 'content': assistant_text})
                if user_text:
                    messages.append({'role': role, 'content': user_text})
                if continue_text:
                    add_continue_message()
                assistant_text = ''
                user_text = ''
                continue_text = ''
                role = 'continue' if line.lower() == 'continue:' else 'continue-chain'
                continue 
            if line.lower() ==  'user:':
                if continue_text:
                    add_continue_message()
                if assistant_text:
                    messages.append({'role': role, 'content': assistant_text})
                if user_text:
                    messages.append({'role': role, 'content': user_text})
                assistant_text = ''
                user_text = ''
                role = 'user'
                continue
            if line.lower() == 'assistant:':
                if continue_text:
                    add_continue_message()
                if user_text:
                    messages.append({'role': role, 'content': user_text})
                    user_text = ''
                if assistant_text:
                    messages.append({'role': role, 'content': assistant_text})
                assistant_text = ''
                role = 'assistant'
                continue
            if role == 'continue' or role == 'continue-chain':
                continue_text += '\n' + line
            elif role == 'assistant':
                assistant_text += '\n' + line
            elif role == 'user':
                user_text += '\n' + line

        if user_text:
            messages.append({'role': role, 'content': user_text})
        if assistant_text:
            messages.append({'role': role, 'content': assistant_text})
        if continue_text:
            add_continue_message()
        message_texts.append(messages)
        if continue_message:
            message_texts += continue_message
            continue_message = []
    chatvision_model.load_models(repo_id=repo_id)
    responses = []
    
    ProgressBar.set_title('[chatvision] - describing the image')
    with torch.no_grad():
        for i, img in enumerate(image):
            ProgressBar.set_progress(i * 100.0/ len(image))
            turn_result = []
            last_response_continue = ''
            last_response = ''
            for msgs in message_texts:
                continue_role = ''
                if msgs[0]['role'] in ('continue', 'continue-chain'):
                    if not last_response_continue:
                        continue
                    continue_content =  msgs[1]['content']
                    if '<to_context:' in continue_content:
                        start = continue_content.index('<to_context:')
                        end = continue_content.index('>', start)
                        if end == -1:
                            print('Error: missing ">" ')
                            break
                        context_name = continue_content[start + 12:end].lower()
                        context_dict[context_name] = last_response_continue
                        continue    
                    continue_role = msgs[0]['role'] != 'continue-chain'
                    msgs[0]['role'] = 'assistant'
                    msgs[0]['content'] = last_response_continue
                if last_response:
                    for msg in msgs:
                        if '<from_assistant>' in msg['content']:
                            msg['content'] = msg['content'].replace('<from_assistant>', last_response)
                        if '<from_context' in msg['content']:
                            start = msg['content'].index('<from_context')
                            end = msg['content'].index('>', start)
                            if end == -1:
                                print('Error: missing ">" ')
                                break
                            context_name = msg['content'][start + 13:end].lower()
                            if context_name in context_dict:
                                msg['content'] = msg['content'][:start] + context_dict[context_name] + msg['content'][end + 1:]
                            else:
                                msg['content'] = msg['content'][:start] + msg['content'][end + 1:]
                                
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
                if not continue_role:
                    last_response_continue = response
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
    