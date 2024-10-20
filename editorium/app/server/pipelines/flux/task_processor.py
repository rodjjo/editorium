from typing import List

import os
import torch
from tqdm import tqdm

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.flux.managed_model import flux_models

SHOULD_STOP = False
PROGRESS_CALLBACK = None  # function(title: str, progress: float)
CURRENT_TITLE = ""


def set_title(title):
    global CURRENT_TITLE
    CURRENT_TITLE = f'CogVideoX: {title}'
    print(CURRENT_TITLE)    


def call_callback(title):
    set_title(title)
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(CURRENT_TITLE, 0.0)


class TqdmUpTo(tqdm):
    def update(self, n=1):
        result = super().update(n)
        if SHOULD_STOP:
            raise StopException("Stopped by user.")
        if PROGRESS_CALLBACK is not None and self.total is not None and self.total > 0:
            PROGRESS_CALLBACK(CURRENT_TITLE, self.n / self.total)
        return result



def generate_flux_image(model_name: str, task_name: str, base_dir: str, input: dict, params: dict):
    flux_models.load_models(model_name)

    result = flux_models.pipe(
        prompt=params['prompt'],
        guidance_scale=params.get('guidance_scale', 0.0),
        height=params.get('height', 768),
        width=params.get('width', 1360),
        num_inference_steps=params.get('num_inference_steps', 4),
        max_sequence_length=params.get('max_sequence_length', 256),
    ).images[0]

    filepath = os.path.join(base_dir, f'{task_name}.png')
    result.save(filepath)
 
    return {
        "output": result,
        "filepath": os.path.join(base_dir, f'{task_name}.png')
    }

    

def process_flux_task(task: dict, callback=None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False
    
    return {
        "success": True,
    }


def cancel_flux_task():
    global SHOULD_STOP
    SHOULD_STOP = True
    return {
        "success": True,
    }


def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False

    return generate_flux_image(
        model_name=config['model_name'],
        task_name=name,
        base_dir=base_dir,
        input=input,
        params=config
    )