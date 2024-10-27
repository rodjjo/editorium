from typing import List

import os
import torch
from tqdm import tqdm
import random
from PIL import Image, ImageFilter 

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.sdxl.managed_model import sdxl_models
from pipelines.common.task_result import TaskResult

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



def generate_sdxl_image(model_name: str, task_name: str, base_dir: str, input: dict, params: dict):
    inpaint_image = input.get('default', {}).get('output', None) or input.get('default', {}).get('result', None)
    inpaint_mask = input.get('mask', {}).get('output', None) or input.get('mask', {}).get('result', None)
    # control_image = input.get('control_image', {}).get('output', None) or input.get('control_image', {}).get('result', None)
    strength = params.get('strength', 0.75)
    if inpaint_mask is not None and inpaint_image is None:
        raise ValueError("It's required a image to inpaint")
    
    if inpaint_mask is not None:
        mode = 'inpaint'
    elif inpaint_image is not None:
        mode = 'img2img'
    else:
        mode = 'txt2img'
        
    inpaint_image = inpaint_image or []
    inpaint_mask = inpaint_mask or []
    if type(inpaint_image) is not list:
        inpaint_image = [inpaint_image]
    
    if type(inpaint_mask) is not list:
        inpaint_mask = [inpaint_mask]
        
    if len(inpaint_mask) > 0 and len(inpaint_image) != len(inpaint_mask):
        raise ValueError("Number of inpaint images and masks must be the same")
    
    if len(inpaint_image) == 0:
        inpaint_image = [None]
        inpaint_mask = [None]
    elif not inpaint_mask:
        inpaint_mask = [None] * len(inpaint_image)
    
    lora_repo_id = params.get('lora_repo_id', '')
    lora_scale = params.get('lora_scale', 1.0)
    sdxl_models.load_models(model_name, mode, lora_repo_id, lora_scale)
    
    add_args = {}
    
    if mode == 'inpaint':
        mask_dilate_size = params.get('mask_dilate_size', 0)
        mask_blur_size = params.get('mask_blur_size', 0)
        kernel_size_dilate = 3
        kernel_size_blur = 3
        if mask_dilate_size > 3:
            kernel_size_dilate = 5
        if mask_blur_size > 3:
            kernel_size_blur = 5
       
        for index, (image, mask) in enumerate(zip(inpaint_image, inpaint_mask)):
            if mask is not None and mask_dilate_size > 0:
                index = 0
                while index < mask_dilate_size:
                    image = image.filter(ImageFilter.MaxFilter(kernel_size_dilate))
                    index += kernel_size_dilate
            if mask is not None and mask_blur_size > 0:
                index = 0
                while index < mask_blur_size:
                    image = image.filter(ImageFilter.GaussianBlur(kernel_size_blur))
                    index += kernel_size_blur
        
        add_args = {
            'image': image, 
            'mask_image': mask,
            **add_args,
        }
    elif mode == 'img2img':
        add_args = {
            'image': inpaint_image,
            'strength': strength,
            **add_args,
        }
    
    seed = params.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
        
    generator = torch.Generator(device='cuda').manual_seed(seed)
    
    result = sdxl_models.pipe(
        prompt=params['prompt'],
        negative_prompt=params.get('negative_prompt', None),
        guidance_scale=params.get('guidance_scale', 5.0),
        height=params.get('height', None),
        width=params.get('width', None),
        num_inference_steps=params.get('num_inference_steps', 50),
        generator=generator,
        **add_args,
    ).images 

    debug_enabled = params.get('globals', {}).get('debug', False)
    if debug_enabled:
        paths = []
        for i, img in enumerate(result):
            filepath = os.path.join(base_dir, f'{task_name}_seed_{seed}_{i}.jpg')
            img.save(filepath)
            paths.append(filepath)
    else:
        paths = [''] * len(result)
 
    return TaskResult(result, paths).to_dict()
    

def process_sdxl_task(task: dict, callback=None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False
    
    return {
        "success": True,
    }


def cancel_sdxl_task():
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

    return generate_sdxl_image(
        model_name=config['model_name'],
        task_name=name,
        base_dir=base_dir,
        input=input,
        params=config
    )