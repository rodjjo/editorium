import torch

import random
from pipelines.omnigen.managed_model import omnigen_models


def generate_omnigen_image(input: dict, params: dict):
    inpaint_image = input.get('default', {}).get('images', None)
    if not inpaint_image:
        inpaint_image = input.get('image', {}).get('images', None)
        
    if not inpaint_image:
        inpaint_image = []

    for i in range(1, 11):
        img = input.get(f'image_{i}', {}).get('images', None)
        if img:
            inpaint_image += img
    
    if inpaint_image is not None:
        mode = 'img2img'
    else:
        mode = 'txt2img'
        
    inpaint_image = inpaint_image or []
    
    if type(inpaint_image) is not list:
        inpaint_image = [inpaint_image]
    
    if len(inpaint_image) == 0:
        inpaint_image = [None]
    
    steps = params.get('steps', 4)
    
    omnigen_models.load_models()

    seed = params.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
        
    generator = torch.Generator(device='cuda').manual_seed(seed)
    
    additional_args = dict(
        prompt=params['prompt'],
        guidance_scale=params.get('cfg', 3.5),
        height=params.get('height', 1024),
        width=params.get('width', 1024),
        num_inference_steps=steps,
        generator=generator,
        use_img_guidance=True,
        img_guidance_scale=1.6,
        max_input_image_size=1024,
        separate_cfg_infer=True,
        offload_model=True,
        use_kv_cache=True,
        offload_kv_cache=True,
        use_input_image_size_as_output=False,
    )
    if mode == 'txt2img':
        result = omnigen_models.pipe(
           **additional_args
        )
    elif mode == 'img2img':
        additional_args = {
            **additional_args,
            'input_images': inpaint_image,
        }
        result = omnigen_models.pipe(
           **additional_args
        )
 
    return {
        'images': result
    }



def process_workflow_task(input: dict, config: dict) -> dict:
    return generate_omnigen_image(
        input=input,
        params=config
    )