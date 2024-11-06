from typing import List

import os
import torch

from pipelines.chatvision.managed_model import chatvision_model


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
    if type(image) is not list:
        image = [image]
    
    msgs = [{'role': 'user', 'content': prompt}]
    
    chatvision_model.load_models(repo_id=repo_id)
    paths = []
    responses = []

    with torch.no_grad():
        for i, img in enumerate(image):
            response = chatvision_model.model.chat(
                image=img,
                msgs=msgs,
                tokenizer=chatvision_model.tokenizer,
                sampling=True, # if sampling=False, beam_search will be used by default
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            responses.append(response)

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
    