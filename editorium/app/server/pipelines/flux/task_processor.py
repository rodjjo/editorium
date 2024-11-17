from typing import List

import os
import torch
import random
from tqdm import tqdm

from PIL import Image, ImageFilter

from pipelines.common.utils import ensure_image
from pipelines.common.exceptions import StopException
from pipelines.common.color_fixer import color_correction
from pipelines.flux.managed_model import flux_models

from task_helpers.progress_bar import ProgressBar

def report(text: str):
    ProgressBar.set_title(f"[FLUX] {text}")


def generate_flux_image(model_name: str, input: dict, params: dict):
    inpaint_image = input.get('default', {}).get('images', None)
    if not inpaint_image:
        inpaint_image = input.get('image', {}).get('images', None)
    inpaint_mask = input.get('mask', {}).get('images', None) 
    control_image = input.get('control_image', {}).get('images', None) 
    
    inpaint_image = ensure_image(inpaint_image)
    inpaint_mask = ensure_image(inpaint_mask)
    control_image = ensure_image(control_image)
    
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
        
    if control_image is not None:
        controlnet_type = params.get('controlnet_type', 'pose')
    else:
        controlnet_type = ''
    
    steps = params.get('steps', 4)
    lora_repo_id = params.get('lora_repo_id', '')
    lora_scale = params.get('lora_scale', 1.0)
    transformer2d_model = params.get('transformer2d_model', None)
    flux_models.load_models(model_name, mode, controlnet_type, lora_repo_id, lora_scale, transformer2d_model)
    flux_models.pipe.scheduler.shift = 3.0 if steps >= 7 else 1.0
    
    if control_image is not None:
        control_args = dict(
            control_image=control_image,
            control_mode=flux_models.control_mode,
            # control_guidance_start=params.get('control_guidance_start', 0.2),
            # control_guidance_end=params.get('control_guidance_end', 0.8),
            controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 1.0),
        )
    else:
        control_args = dict()
    
    seed = params.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
        
    generator = torch.Generator(device='cuda').manual_seed(seed)
    additional_args = dict(
        prompt=params['prompt'],
        prompt_2=params['prompt'],
        guidance_scale=params.get('cfg', 3.5),
        height=params.get('height', 768),
        width=params.get('width', 1360),
        num_inference_steps=steps,
        max_sequence_length=params.get('max_sequence_length', 256),
        generator=generator,
        **control_args,
    )
    
    report(f"Generating image with model {transformer2d_model if transformer2d_model else model_name}")
    
    if mode == 'txt2img':
        result = flux_models.pipe(
           **additional_args
        ).images
    elif mode == 'img2img':
        all_results = []
        for image in inpaint_image:
            additional_args = {
                **additional_args,
                'image': inpaint_image,
                'strength': strength,
            }
            result = flux_models.pipe(
            **additional_args
            ).images
            all_results += result
        result = all_results
    else: # mode == 'inpaint'
        mask_dilate_size = params.get('mask_dilate_size', 0)
        mask_blur_size = params.get('mask_blur_size', 0)
        kernel_size_dilate = 3
        kernel_size_blur = 3
        if mask_dilate_size > 3:
            kernel_size_dilate = 5
        if mask_blur_size > 3:
            kernel_size_blur = 5

        all_results = []
        for index, (image, mask) in enumerate(zip(inpaint_image, inpaint_mask)):
            if mask is not None and mask_dilate_size > 0:
                index = 0
                while index < mask_dilate_size:
                    mask = mask.filter(ImageFilter.MaxFilter(kernel_size_dilate))
                    index += kernel_size_dilate
            if mask is not None and mask_blur_size > 0:
                index = 0
                while index < mask_blur_size:
                    mask = mask.filter(ImageFilter.GaussianBlur(kernel_size_blur))
                    index += kernel_size_blur

            additional_args = {
                **additional_args,
                'image': image,
                'mask_image': mask,
                'strength': strength,
            }
            results = flux_models.pipe(
                **additional_args
            ).images
        
            correct_colors = params.get('correct_colors', False)
            if mask:
                for i, result in enumerate(results):
                    mask = mask.convert("RGBA")
                    mask.putalpha(mask.split()[0])
                    result = result.resize(image.size)
                    try:
                        if correct_colors:
                            results[i] = color_correction(results[i], image)
                        results[i] = Image.composite(result, image, mask)
                    except:
                        print(f"\n\n!!!\n\n {input} - image: {image} - mask: {mask} - result: {result}")
                        raise
            all_results += results
        result = all_results
 
    return {
        'images': result
    }

    

def process_flux_task(task: dict) -> dict:
    return {
        "success": True,
    }


def process_workflow_task(input: dict, config: dict) -> dict:
    return generate_flux_image(
        model_name=config['model_name'],
        input=input,
        params=config
    )

    
def process_workflow_list_model(list_loras: bool):
    return flux_models.list_models(list_loras)

