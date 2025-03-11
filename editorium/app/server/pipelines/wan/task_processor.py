
import traceback
import os
import sys
import random
import subprocess
from typing import Literal, List
import torch
from diffusers.utils import export_to_video, load_video
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from pipelines.common import utils
from pipelines.common.rife_model import rife_inference_with_latents
from pipelines.common.prompt_parser import iterate_prompts
from pipelines.wan.managed_model import wan21_model
from pipelines.common.exceptions import StopException
from task_helpers.progress_bar import ProgressBar

SHOULD_STOP = False
PROGRESS_CALLBACK = None  # function(title: str, progress: float)


def set_title(title):
    ProgressBar.set_title(f'Wan 2.1 Video: {title}')
    ProgressBar.set_progress(0.0)


def generate_video(
    prompt: str,
    negative_prompt: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    quant: bool = False,
    should_use_14b_model: bool = False,
    should_upscale: bool = False,
    strength: float = 0.8,
):
    image = None
    video = None
    
    wan21_model.load_models(
        should_use_14b_model=should_use_14b_model,
        pipeline_type=generate_type,
        lora_repo_id=lora_path,
        lora_scale=lora_rank,
    )
    
    if seed == -1:
        seed = random.randint(0, 1000000)

    if num_videos_per_prompt > 1:
        prompt = [prompt] * num_videos_per_prompt
        generator = []
        for index in range(num_videos_per_prompt):
            generator.append(torch.Generator(device="cuda").manual_seed(seed + index))
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_videos_per_prompt": 1, 
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "output_type": "pil",
        "max_sequence_length": 512,
    }
    
    if generate_type == "i2v":
        if type(image_or_video_path) == str:
            image = utils.load_image_rgb(image_or_video_path.strip())
        else:
            image = image_or_video_path
        pipe_args["image"] = image
        pipe_args["num_frames"] = 81
    else:
        pipe_args["num_frames"] = 81
        

    wan21_model.pipe.progress_bar = lambda total: ProgressBar(total=total)

    set_title("Generating video")
    
    video_generate = wan21_model.pipe(**pipe_args).frames

    if len(video_generate) < 1:
        set_title("No video generated.")
        return []

    videos = []
    if num_videos_per_prompt < 2:
        videos = [video_generate[0]]
    else:
        for i in range(len(video_generate)):
            videos.append(video_generate[i])
    return videos


def process_workflow_task(input: dict, config: dict):
    input_images = input.get('default', {}).get('images', None)
    if not input_images:
        input_images = input.get('image', {}).get('images', None)
    input_videos = input.get('video', {}).get('images', None)
            
    if input_images and input_videos:
        raise ValueError("Both images and videos were passed as input")
    
    if input_images is not None:
        generate_type = 'i2v'
    elif input_videos is not None:
        generate_type = 'v2v'
    else:
        generate_type = 't2v'

    input_param = input_images if input_images else input_videos
    
    videos = []
    if not input_param:
        input_param = [None]
    
    for index, data in enumerate(input_param):
        videos += generate_video(
            prompt=config['prompt'],
            negative_prompt=config.get('negative_prompt', None),
            lora_path=config.get('lora_path', None),
            lora_rank=config.get('lora_rank', 128),
            image_or_video_path=data,
            num_inference_steps=config.get('num_inference_steps', 50),
            guidance_scale=config.get('guidance_scale', 6.0),
            num_videos_per_prompt=config.get('num_videos_per_prompt', 1),
            generate_type=generate_type,
            seed=config.get('seed', -1),
            quant=config.get('quant', False),
            should_upscale=config.get('should_upscale', False),
            strength=config.get('strength', 0.8),
        )
        
    assert len(videos) > 0, "No videos generated"
    assert type(videos[0]) == list, "Videos should be a list of images" + str(type(videos[0]))
    assert type(videos[0][0]) == Image.Image, "Videos should be a list of images" + str(type(videos[0][0]))
        
    images = []
    for v in videos:
        for img in v:
            images.append(img)

    return {
        "videos": videos,
        "images": images,
    }
