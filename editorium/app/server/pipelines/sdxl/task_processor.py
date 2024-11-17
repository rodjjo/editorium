from typing import List

import os
import torch
from tqdm import tqdm
import random
from PIL import Image, ImageFilter 

from pipelines.sdxl.managed_model import sdxl_models
from pipelines.common.color_fixer import color_correction
from pipelines.common.utils import ensure_image
from task_helpers.progress_bar import ProgressBar


def report(text: str):
    ProgressBar.set_title(f"[SDXL] {text}")


def generate_sdxl_image(model_name: str, input: dict, params: dict):
    inpaint_image = input.get('default', {}).get('images', None)
    if not inpaint_image:
        inpaint_image = input.get('image', {}).get('images', None)
    inpaint_mask = input.get('mask', {}).get('images', None) 
    control_image = input.get('control_image', {}).get('images', None)
    strength = params.get('strength', 0.75)
    if inpaint_mask is not None and inpaint_image is None:
        raise ValueError("It's required a image to inpaint")
    
    inpaint_image = ensure_image(inpaint_image)
    inpaint_mask = ensure_image(inpaint_mask)
    control_image = ensure_image(control_image)
    
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
        
    inpaint_image = [i.convert("RGB") if i and i.mode != 'RGB' else i for i in inpaint_image]
    
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
    for adapter_index in range(1, 3):
        param_name = f'adapter_{adapter_index}'
        if param_name not in input:
            continue
        adapter = input.get(param_name, {}).get('data', {})
        if not adapter:
            raise ValueError(f"Adapter {adapter_index} not found")
        adapter_models.append(
            adapter['adapter_model']
        )
        adapter_scale.append(params.get(f'ip_adapter_scale_{adapter_index}', 0.6))
        image = ensure_image(adapter['image'])
        if type(image) is not list:
            image = [image]
        if inpaint_image is not None and len(inpaint_image) > 0 and len(inpaint_image) != len(image):
            if len(inpaint_image) == 1:
                image = [image[0]] * len(inpaint_image)
            else:
                raise ValueError("Number of controlnet images must be the same as inpaint images")
        image = [i.resize((224, 224)) for i in image]
        adapter_images.extend(image)

    if len(adapter_models) == 0:
        adapter_models = [None]
        
    sdxl_models.load_models(
        model_name, mode, lora_repo_id, lora_scale, 
        unet_model, controlnet_type, adapter_models[0], 
        load_state_dict=params.get('load_state_dict', False),
    )
    
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
            
    seed = params.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
        
    generator = torch.Generator(device='cuda').manual_seed(seed)
    
    cfg = params.get('cfg', 5.0)
    if cfg == 0.0:
        negative_prompt = ''
    else:
        negative_prompt = params.get('negative_prompt', '')
    
    if hasattr(sdxl_models.pipe, 'generate'):
        pipe = lambda *args, **kwargs:  sdxl_models.pipe.generate(*args, **kwargs) 
    else:
        pipe = lambda *args, **kwargs: sdxl_models.pipe(*args, **kwargs).images 
    
    report(f"Generating image with model {unet_model if unet_model else model_name}")
    
    if mode == 'inpaint':
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
            results = pipe(
                prompt=params['prompt'],
                negative_prompt=negative_prompt,
                guidance_scale=cfg,
                height=params.get('height', None),
                width=params.get('width', None),
                num_inference_steps=params.get('steps', 50),
                generator=generator,
                image=image,
                mask_image=mask,
                strength= strength,
                **add_args,
            )
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
    elif mode == 'img2img':
        all_results = []
        for  image in inpaint_image:
            result = pipe(
                prompt=params['prompt'],
                negative_prompt=negative_prompt,
                guidance_scale=cfg,
                height=params.get('height', None),
                width=params.get('width', None),
                num_inference_steps=params.get('steps', 50),
                generator=generator,
                image= inpaint_image,
                strength= strength,
                **add_args,
            )
            all_results += result
        result = all_results
    else:
        result = pipe(
            prompt=params['prompt'],
            negative_prompt=negative_prompt,
            guidance_scale=cfg,
            height=params.get('height', None),
            width=params.get('width', None),
            num_inference_steps=params.get('steps', 50),
            generator=generator,
            **add_args,
        )


    return {
        'images': result
    }
    

def process_workflow_task(input: dict, config: dict) -> dict:
    return generate_sdxl_image(
        model_name=config['model_name'],
        input=input,
        params=config
    )

    
def process_workflow_list_model(list_loras: bool):
    return sdxl_models.list_models(list_loras)

