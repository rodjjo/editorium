from typing import List

import os
import re
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.common.task_result import TaskResult
from pipelines.sd15.managed_model import sd15_models
from pipelines.sd15.loader import LORA_DIR



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



def get_lora_path(lora: str) -> str:
    for e in ('.safetensors', '.ckpt'):
        filepath = os.path.join(LORA_DIR, f'{lora}{e}')
        if os.path.exists(filepath):
            return filepath
    return None


def parse_prompt_loras(prompt: str):
    lora_re = re.compile('<lora:[^:]+:[^>]+>')
    lora_list = re.findall(lora_re, prompt)

    lora_items = []
    for lora in lora_list:
        lora = lora.replace('<', '').replace('>', '')
        p = lora.split(':')
        if len(p) != 3:
            continue
        p = [p[1], p[2]]
        try:
            weight = float(p[1])
        except Exception:
            print(f"Invalid lora weigth {p[1]}")
            continue
        filepath = get_lora_path(p[0])
        if not filepath:
            print(f"Lora not found: {p[0]}")
            continue
        lora_items.append([filepath, weight])
    return re.sub(lora_re, '', prompt), lora_items


def randn(seed, shape):
    torch.manual_seed(seed)
    return torch.randn(shape, device='cuda')


def randn_without_seed(shape):
    return torch.randn(shape, device='cuda')

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val) * omega)/so).unsqueeze(1) * low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def create_latents_noise(shape, seed, subseed=None, subseed_strength=0.0):
    xs = []
    subnoise = None
    if subseed is not None:
        subnoise = randn(subseed, shape)
    # randn results depend on device; gpu and cpu get different results for same seed;
    # the way I see it, it's better to do this on CPU, so that everyone gets same result;
    # but the original script had it like this, so I do not dare change it for now because
    # it will break everyone's seeds.
    noise = randn(seed, shape)

    if subnoise is not None:
        noise = slerp(subseed_strength, noise, subnoise)

    xs.append(noise)
    x = torch.stack(xs).to('cuda')
    return x


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

@torch.no_grad()
def run_pipeline(
        pipeline: object,
        prompt: str,
        negative: str = None,
        seed: int = -1,
        cfg: float = 7.5,
        steps: int = 20,
        width: int = 512,
        height: int = 512,
        strength: float = 0.75,
        batch_size: int = 1,
        input_image: Image.Image = None,
        input_mask: Image.Image = None,
        inpaint_mode: str = "original",
        controlnets: list = [],
        use_float16: bool = True):
    
    pipeline_type = "txt2img"
    if input_image is not None:
        pipeline_type = "img2img"
        if input_mask is not None:
            pipeline_type = "inpaint2img"
    elif input_mask is not None:
        raise ValueError("input_mask provided without input_image")

    device = 'cuda'

    if len(negative or '') < 2:
        negative = None

    if steps > 50:
        steps = 50
    
    usefp16 = { 
        True: torch.float16,
        False: torch.float32
    }
    dtype = usefp16[use_float16]

    if 'img2img' in pipeline_type  and input_mask is not None:
        pipeline_type = pipeline_type.replace('img2img', 'inpaint2img')

    if 'inpaint2img' in pipeline_type and  inpaint_mode == "img2img":
        pipeline_type = pipeline_type.replace('inpaint2img', 'img2img')

    if width % 8 != 0:
        width += 8 - width % 8

    if height % 8 != 0:
        height += 8 - height % 8

    variation_enabled = False
    var_stren = 0
    subseed = None
    
    shape = (4, height // 8, width // 8 )
    latents_noise = create_latents_noise(shape, seed, subseed, var_stren)
    latents_noise = latents_noise.to(dtype=dtype)
    
    generator = [
        torch.Generator(device=device).manual_seed(seed + i)
        for i in range(batch_size)
    ]

    
    print("started")

    print("creating the pipeline")
    leditpp = bool(input_image) and pipeline_type == 'txt2img'
    
        
    print("pipeline created")

    def progress_preview(step, timestep, latents):
        # TODO: Implement progress callback
        pass

    additional_args = {
        'generator': generator
    }
    if pipeline_type == 'txt2img':
        additional_args = {
            'width': width, 
            'height': height,
            'latents': latents_noise,
        }
        if len(controlnets):
            images = []
            conds = []
            for c in controlnets:
                if c['strength'] < 0:
                    c['strength'] = 0
                if c['strength'] > 2.0:
                    c['strength'] = 2.0
                images.append(c['image'])
                conds.append(c['strength'])
            if len(images) == 1:
                images = images[0]
            if len(conds) == 1:
                conds = conds[0]
            additional_args['image'] = images
            additional_args['controlnet_conditioning_scale'] = conds
    elif pipeline_type == 'img2img':
        additional_args = {
            'image': input_image,
            'strength': strength,
        }
        if len(controlnets):
            additional_args['width'] = width
            additional_args['height'] = height
            images = []
            conds = []
            for c in controlnets:
                if c['strength'] < 0:
                    c['strength'] = 0
                if c['strength'] > 2.0:
                    c['strength'] = 2.0
                images.append(c['image'])
                conds.append(c['strength'])
            if len(images) == 1:
                images = images[0]
            if len(conds) == 1:
                conds = conds[0]
            additional_args['controlnet_conditioning_image'] = images
            additional_args['controlnet_conditioning_scale'] = conds
    elif pipeline_type == 'inpaint2img':
        '''
        if not ):
            return [{
                "error": "The current model is not for in painting"
            }]
        '''
        image = input_image
        mask = input_mask

        additional_args = {
            'image': image,
            'mask_image': mask,
            'width': width,
            'height': height,
            'latents': latents_noise,
        }
        if len(controlnets):
            images = []
            conds = []
            has_inpaint = False
            for c in controlnets:
                if c['strength'] < 0:
                    c['strength'] = 0
                if c['strength'] > 2.0:
                    c['strength'] = 2.0
                if c['mode'] == 'inpaint':
                    images.append(make_inpaint_condition(image, mask))
                    has_inpaint = True
                else:
                    images.append(c['image'])
                conds.append(c['strength'])
            if len(images) == 1:
                images = images[0]
            if len(conds) == 1:
                conds = conds[0]
            if has_inpaint:
                additional_args['control_image'] = images
                additional_args['controlnet_conditioning_scale'] = conds
            else:
                additional_args['controlnet_conditioning_image'] = images
                additional_args['controlnet_conditioning_scale'] = conds


    latents_noise.to(device)
    with torch.inference_mode(), torch.autocast(device):
        additional_args['callback'] = progress_preview
        additional_args['callback_steps'] = 1

        if len(controlnets):
            batch_size = 1

        if type(pipeline.scheduler).__name__ == 'LCMScheduler':
            if pipeline_type == 'txt2img':
                if cfg > 2:
                    cfg = 2
            elif pipeline_type == 'inpaint2img':
                if cfg > 4:
                    cfg = 4
            else:
                if cfg > 1:
                    cfg = 1

            if steps > 8:
                steps = 8

        if  batch_size > 1 and additional_args.get('latents') is not None:
            del additional_args['latents']

        if batch_size > 1:
            additional_args['batch_size'] = batch_size
            additional_args['num_images_per_prompt'] = batch_size
        
        if hasattr(pipeline, 'load_ip_adapter'):
            image_list_adapt = []
            #if face:
            #    image_list_adapt  += [face]

            # if adapter_image:
            #    image_list_adapt += [adapter_image]
            if image_list_adapt:
                additional_args["ip_adapter_image"] = image_list_adapt

        
        
        additional_args['guidance_scale'] = cfg
        additional_args['num_inference_steps'] = steps
        additional_args['prompt'] = prompt
        additional_args['negative_prompt'] = negative
        
        print("generating the variation" if variation_enabled else "generating the image")
        result = pipeline(
            **additional_args,
        ).images 
   
    print("image generated")
    return [r for r in result]


def generate_sd15_image(model_name: str, task_name: str, base_dir: str, input: dict, params: dict):
    prompt, lora_list = parse_prompt_loras(params['prompt'])
    negative_prompt = params.get('negative_prompt', None)
    
    inpaint_image = input.get('default', {}).get('output', None)
    inpaint_mask = input.get('mask', {}).get('output', None)
    
    sd15_models.load_models(
        model_name, 
        inpainting_mode=inpaint_mask is not None,
        image2image=inpaint_image is not None,
        lora_list=lora_list,
        use_lcm=False,
        scheduler_name='EulerAncestralDiscreteScheduler',
        use_float16=True,
    )
    
    seed = params.get('seed', -1)
    if seed != -1:
        seed = random.randint(0, 1000000)

    results = run_pipeline(
        pipeline=sd15_models.pipe,
        prompt=prompt,
        negative=negative_prompt,
        seed=seed,
        cfg=params.get('cfg', 7.5),
        steps=params.get('steps', 20),
        width=params.get('width', 512),
        height=params.get('height', 512),
        strength=params.get('strength', 0.75),
        batch_size=params.get('batch_size', 1),
        input_image=inpaint_image,
        input_mask=inpaint_mask,
        inpaint_mode="original",
        controlnets=params.get('controlnets', []),
        use_float16=True
    )

    filepath = os.path.join(base_dir, f'{task_name}_seed_{seed}.png')
    paths = []
    for i, result in enumerate(results):
        path2save = filepath.replace('.png', f'_{i}.png')
        result.save(path2save)
        paths.append(path2save)
 
    return TaskResult(results, paths).to_dict()


def process_sd15_task(task: dict, callback=None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False
    
    return {
        "success": True,
    }


def cancel_sd15_task():
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

    return generate_sd15_image(
        model_name=config['model_name'],
        task_name=name,
        base_dir=base_dir,
        input=input,
        params=config
    )