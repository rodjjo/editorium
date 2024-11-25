from typing import List

import os
import torch
import random
from tqdm import tqdm

from PIL import Image, ImageFilter

from pipelines.common.exceptions import StopException
from pipelines.common.utils import ensure_image
from pipelines.common.color_fixer import color_correction
from pipelines.sd35.managed_model import sd35_models
from task_helpers.progress_bar import ProgressBar


def report(text: str):
    ProgressBar.set_title(f"[SD35] {text}")


def generate_sd35_image(model_name: str, input: dict, params: dict):
    inpaint_image = input.get('default', {}).get('images', None)
    if not inpaint_image:
        inpaint_image = input.get('image', {}).get('images', None)
    inpaint_mask = input.get('mask', {}).get('images', None) 
    
    inpaint_image = ensure_image(inpaint_image)
    inpaint_mask = ensure_image(inpaint_mask)
    
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
        
    width = params.get('width', 1360)    
    height = params.get('height', 768)
    
    if inpaint_image and inpaint_image[0]:
        width = inpaint_image[0].width
        height = inpaint_image[0].height
        
    original_width = width
    original_height = height
    
    if width % 64 != 0:
        width += 64 - width % 64

    if height % 64 != 0:
        height += 64 - height % 64
        
    if width != original_width or height != original_height:
        if inpaint_image:
            for i, img in enumerate(inpaint_image):
                if not img:
                    continue
                image = Image.new('RGB', (width, height), (255, 255, 255))
                image.paste(img, (0, 0))
                inpaint_image[i] = image
        if inpaint_mask:
            for i, img in enumerate(inpaint_mask):
                if not img:
                    continue
                image = Image.new('RGB', (width, height), (255, 255, 255))
                image.paste(img, (0, 0))
                inpaint_mask[i] = image
    
    steps = params.get('steps', 4)
    lora_repo_id = params.get('lora_repo_id', '')
    lora_scale = params.get('lora_scale', 1.0)
    transformer2d_model = params.get('transformer2d_model', None)
    sd35_models.load_models(model_name, mode, lora_repo_id, lora_scale, transformer2d_model)
    sd35_models.pipe.scheduler.shift = 3.0 if steps >= 7 else 1.0
    

    seed = params.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
        
    generator = torch.Generator(device='cuda').manual_seed(seed)
    additional_args = dict(
        prompt=params['prompt'],
        prompt_2=params['prompt'],
        guidance_scale=params.get('cfg', 3.5),
        num_inference_steps=steps,
        max_sequence_length=params.get('max_sequence_length', 256),
        generator=generator
    )
    
    report(f"Generating image with model {transformer2d_model if transformer2d_model else model_name}")

    if mode == 'txt2img':
        result = sd35_models.pipe(
            height=height,
            width=width,
           **additional_args
        ).images
    elif mode == 'img2img':
        all_results = []
        for image in inpaint_image:
            result = sd35_models.pipe(
                image=image,
                strength=strength,
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
            
            results = sd35_models.pipe(
                **additional_args
            ).images
            
            correct_colors = params.get('correct_colors', False)
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

    for i, image in enumerate(result):
        if image.width != original_width or image.height != original_height:
            # get a croped region from 0, 0 to original_width, original_height
            result[i] = image.crop((0, 0, original_width, original_height))

    return {
        'images': result
    }


def process_workflow_task(input: dict, config: dict) -> dict:
    return generate_sd35_image(
        model_name=config['model_name'],
        input=input,
        params=config
    )

    
def process_workflow_list_model(list_loras: bool):
    return sd35_models.list_models(list_loras)


