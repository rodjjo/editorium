
import traceback
import os
import sys
import random
import subprocess
from typing import Literal, List, Optional, Union
import torch
from diffusers.utils import export_to_video, load_video
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from pipelines.common import utils
from pipelines.common.rife_model import rife_inference_with_latents
from pipelines.common.prompt_parser import iterate_prompts
from pipelines.ltx.modules.pipelines.pipeline_ltx_video import ConditioningItem
from pipelines.ltx.modules.utils.skip_layer_strategy import SkipLayerStrategy
from pipelines.ltx.managed_model import ltx_model
from pipelines.common.exceptions import StopException
from task_helpers.progress_bar import ProgressBar

SHOULD_STOP = False
PROGRESS_CALLBACK = None  # function(title: str, progress: float)


def set_title(title):
    ProgressBar.set_title(f'Wan 2.1 Video: {title}')
    ProgressBar.set_progress(0.0)
    

def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
) -> torch.Tensor:
    """Load and process an image into a tensor.

    Args:
        image_input: Either a file path (str) or a PIL Image object
        target_height: Desired height of output tensor
        target_width: Desired width of output tensor
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def prepare_conditioning(
    conditioning_images: list[Image.Image],
    conditioning_strengths: List[float],
    conditioning_start_frames: List[int],
    height: int,
    width: int,
    num_frames: int,
    padding: tuple[int, int, int, int],
    pipeline: object,
) -> Optional[List[ConditioningItem]]:
    if len(conditioning_images) != len(conditioning_strengths) or \
            len(conditioning_images) != len(conditioning_start_frames):
        return []

    conditioning_items = []
    for image, strength, start_frame in zip(
        conditioning_images, conditioning_strengths, conditioning_start_frames
    ):
    
        frame_tensor = load_image_to_tensor_with_resize_and_crop(
            image, height, width
        )
        frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
        conditioning_items.append(
            ConditioningItem(frame_tensor, start_frame, strength)
        )

    return conditioning_items

def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


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
    width: int = 704,
    height: int = 480,
    num_frames: int = 121,
    frame_rate: int = 25,
    should_upscale: bool = False,
    strength: float = 0.8,
    stg_skip_layers: str = "19",
    stg_mode: str = "attention_values",
    stg_scale: float = 1.0,
    stg_rescale: float = 0.7,
    image_cond_noise_scale: float = 0.15,
    decode_timestep: float = 0.025,
    decode_noise_scale: float = 0.0125,
    precision: str = "bfloat16",
    offload_to_cpu: bool = True,
    device: str = None,
):
    image = None
    video = None
    
    ltx_model.load_models(
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
    
    if generate_type == "i2v":
        start_frame = [0]
        conditioning_images = [image_or_video_path]
        conditioning_strengths = [1.0]
    else:
        start_frame = []
        conditioning_images = []
        conditioning_strengths = []
        
    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    conditioning_items = (
        prepare_conditioning(
            conditioning_images=conditioning_images,
            conditioning_strengths=conditioning_strengths,
            conditioning_start_frames=start_frame,
            height=height,
            width=width,
            num_frames=num_frames,
            padding=padding,
            pipeline=ltx_model.pipe,
        )
        if conditioning_images
        else None
    )        
    
    # Set spatiotemporal guidance
    skip_block_list = [int(x.strip()) for x in stg_skip_layers.split(",")]
    
    if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
    }
    
        
    pipe_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_attention_mask": None,
        "negative_prompt_attention_mask": None,
        "skip_layer_strategy": skip_layer_strategy,
        "skip_block_list": skip_block_list,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_images_per_prompt": 1, 
        "height": height_padded,
        "width": width_padded,
        "num_frames": num_frames_padded,
        "frame_rate": frame_rate,
        "stg_scale": stg_scale,
        "do_rescaling": stg_rescale != 1,
        "rescaling_scale": stg_rescale,
        "output_type": "pt",
        "callback_on_step_end": None,
        "conditioning_items": conditioning_items,
        "is_video": True,
        "vae_per_channel_normalize": True,
        "image_cond_noise_scale": image_cond_noise_scale,
        "decode_timestep": decode_timestep,
        "decode_noise_scale": decode_noise_scale,
        "mixed_precision": (precision == "mixed_precision"),
        "offload_to_cpu": offload_to_cpu,
        "device": device,
        "enhance_prompt": False,
    }

    ltx_model.pipe.progress_bar = lambda total: ProgressBar(total=total)

    set_title("Generating video")
    
    video_generate = ltx_model.pipe(**pipe_args).images
    
    # Crop the padded images to the desired resolution and number of frames
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = video_generate.shape[3]
    if pad_right == 0:
        pad_right = video_generate.shape[4]
    video_generate = video_generate[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
    
    videos = []
    for i in range(video_generate.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = video_generate[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = frame_rate
        height, width = video_np.shape[1:3]
        if video_np.shape[0] == 1:
            videos.append([Image.fromarray(video_np[0])])
        else:
            videos.append([Image.fromarray(frame) for frame in video_np])    
    
    video_generate = videos
    
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
