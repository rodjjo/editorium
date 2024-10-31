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
    control_image = input.get('control_image', {}).get('output', None) or input.get('control_image', {}).get('result', None)
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
    unet_model = params.get('unet_model', None)
    
    if control_image is None:
        controlnet_type = ''
    else:
        controlnet_type = params.get('controlnet_type', 'pose')
        
    adapter_images = []
    adapter_models = []
    adapter_scale  = []
    for adapter_index in range(1, 2):
        param_name = f'adapter_{adapter_index}'
        if param_name not in input:
            continue
        adapter = input.get(param_name, {}).get('default', {})
        if not adapter:
            raise ValueError(f"Adapter {adapter_index} not found")
        adapter_models.append(
            adapter['adapter_model']
        )
        adapter_scale.append(params.get(f'ip_adapter_scale_{adapter_index}', 0.6))
        image = adapter['image']
        if type(image) is str:
            image = [Image.open(image)]
        elif type(image) is list:
            image = [Image.open(i) if type(i) is str else i for i in image]
        if inpaint_image is not None and len(inpaint_image) > 0 and len(inpaint_image) != len(image):
            if len(inpaint_image) == 1:
                image = [image[0]] * len(inpaint_image)
            else:
                raise ValueError("Number of controlnet images must be the same as inpaint images")
        image = [i.resize((224, 224)) for i in image]
        adapter_images.extend(image)

    if len(adapter_models) == 0:
        adapter_models = [None]
        
    sdxl_models.load_models(model_name, mode, lora_repo_id, lora_scale, unet_model, controlnet_type, adapter_models[0])
    
    add_args = {}
    
    if adapter_images:
        add_args['pil_image'] = adapter_images[0]
        add_args['scale'] = adapter_scale[0]
    
    if control_image is not None:
        # control_mode=sdxl_models.control_mode,
        add_args['controlnet_conditioning_scale'] = params.get('controlnet_conditioning_scale', 1.0)
        if mode == 'txt2img':
            add_args['image'] = control_image
        else:
            add_args['control_image'] = control_image
    
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
            'strength': strength,
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
    
    pipe_args = dict(
        prompt=params['prompt'],
        negative_prompt=params.get('negative_prompt', None),
        guidance_scale=params.get('cfg', 5.0),
        height=params.get('height', None),
        width=params.get('width', None),
        num_inference_steps=params.get('steps', 50),
        generator=generator,
        **add_args,
    )
    if hasattr(sdxl_models.pipe, 'generate'):
        result = sdxl_models.pipe.generate(**pipe_args) 
    else:
        result = sdxl_models.pipe(**pipe_args).images 

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