from typing import List, Dict

import gc
import os
import json
import torch
import random
from transformers import set_seed, TextStreamer, StoppingCriteria, StoppingCriteriaList

from pipelines.chatstory.managed_model import chatstory_model
from task_helpers.progress_bar import ProgressBar
from pipelines.common.utils import ensure_image

class CanceledChecker:
    def __call__(self, *args, **kwargs) -> bool:
        return False # TODO return True if the task should be canceled



class GenerationStopper(StoppingCriteria):
    def __init__(self, stop_tokens: dict[str, list[int | list[int]]]):
        self.stop_token_ids = []
        for t in stop_tokens.values():
            if any(isinstance(x, list) for x in t):  # if t is nested list
                for x in t:
                    self.stop_token_ids.append(torch.tensor(x))
            else:
                self.stop_token_ids.append(torch.tensor(t))
            assert isinstance(t, list) or isinstance(t, int)
        self.stop_token_words = stop_tokens.keys()

    def __repr__(self):
        return f"Stopping words: {self.stop_token_words}"

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for t in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(t) :].to("cpu"), t).all():
                return True
        return False

    @property
    def criteria(self):
        return StoppingCriteriaList([self])

    def format(self, sentence: str):
        for w in self.stop_token_words:
            if w in sentence[-len(w) :]:
                sentence = sentence[: -len(w)]
        return sentence


def create_stop_criteria(tokenizer):
    stop_words = ['<|im_end|>', '<|endoftext|>', '|im_end|']
    stop_words_ids = [tokenizer(stop_word, add_special_tokens=False)['input_ids'] for stop_word in stop_words]
    gen_stop = GenerationStopper({stop_word: stop_word_id for stop_word, stop_word_id in zip(stop_words, stop_words_ids)})
    return gen_stop.criteria


def generate_text(repo_id: str, 
                  prompt: str,
                  system_prompt: str = '',
                  temperature: float = 0.7,
                  globals: dict = {}) -> dict:
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
    chatstory_model.load_models(repo_id=repo_id)
    responses = []
    
    streamer = TextStreamer(chatstory_model.tokenizer, skip_prompt=True)

    ProgressBar.set_title('[chatstory] - processing the chat')
    with torch.no_grad():
        msgs = []
        for msg_txt in message_texts:
            msgs.append([m for m in msg_txt if m['content'].strip()])
        message_texts = msgs
        turn_result = []
        last_response = ''
        for msgs in message_texts:
            should_skip = False
            for msg in msgs:
                if '<noimg>' in msg['content']:
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
            
            torch.manual_seed(random.randint(0, 100000))
            set_seed(random.randint(0, 100000))
            
            msgs = [{
                'role': 'system',
                'content': system_prompt
            }] + msgs

            response = chatstory_model.pipe(
                msgs,
                max_new_tokens=4096,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
                #stopping_criteria=stopping_criteria
                # eos_token_id=chatstory_model.tokenizer.eos_token_id,
            )

            response = response[0]["generated_text"][-1]['content']
            
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
    gc.collect()
    torch.cuda.empty_cache()
    return generate_text(
        repo_id=config['repo_id'],
        prompt=config['prompt'],
        system_prompt=config.get('system_prompt', ''),
        temperature=config.get('temperature', 0.7),
        globals=config.get('globals', {})
    )
    