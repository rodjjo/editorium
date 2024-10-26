from typing import List

import os
import torch

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.common.task_result import TaskResult
from pipelines.chatvision.managed_model import chatvision_model


class CanceledChecker:
    def __call__(self, *args, **kwargs) -> bool:
        return False # TODO return True if the task should be canceled


def generate_text(base_dir: str,
                  name: str,
                  repo_id: str, 
                  image,
                  prompt: str,
                  system_prompt: str = '',
                  temperature: float = 0.7,
                  callback: callable = None) -> dict:
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

            text_path = os.path.join(base_dir, f'{name}_{i}.txt')
            with open(text_path, 'w') as f:
                f.write("# system_prompt: \n")
                f.write(system_prompt)
                f.write("\n\n# prompt: \n")
                f.write(prompt)
                f.write("\n\n# response: \n")
                f.write(response)
            paths.append(text_path)
            responses.append(response)

    return {
        "default": responses,
        "filepath": paths,
    }
    

def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False
    
    image = input.get('image', {}).get('output', None) or input.get('image', {}).get('result', None)
    
    return generate_text(
        base_dir=base_dir,
        name=name,
        repo_id=config['repo_id'],
        image=image,
        prompt=config['prompt'],
        system_prompt=config.get('system_prompt', ''),
        temperature=config.get('temperature', 0.7),
        callback=callback
    )
    