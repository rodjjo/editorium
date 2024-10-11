
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
import pipelines.cogvideo.utils as utils
from pipelines.cogvideo.rife_model import rife_inference_with_latents


from tqdm import tqdm
import numpy as np


from pipelines.cogvideo.cogvideox_lora_trainer import train_lora_model
from pipelines.cogvideo.managed_model import cogvideo_model

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

class StopException(Exception):
    pass

class TqdmUpTo(tqdm):
    def update(self, n=1):
        result = super().update(n)
        if SHOULD_STOP:
            raise StopException("Stopped by user.")
        if PROGRESS_CALLBACK is not None and self.total is not None and self.total > 0:
            PROGRESS_CALLBACK(CURRENT_TITLE, self.n / self.total)
        return result


def load_image_with_pil(image_path: str):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_pil_image(image: Image, hd=False):
    if hd:
        size = (1072, 720)
    else:
        size = (736, 496)
    return image.resize(size)


to_tensors_transform = transforms.ToTensor()
to_pil_transform = transforms.ToPILImage()

blank_image = Image.new("RGB", (736, 496), (0, 0, 0))

def get_non_existing_path(output_path: str) -> str:
    file_index = 0
    saved_path = output_path.replace(".mp4", "")
    while os.path.exists(output_path):
        output_path = f"{saved_path}_{file_index}.mp4"
        file_index += 1
    return output_path


def save_video(frames, output_path, should_upscale=False):
    if not os.path.exists("/app/output_dir/output/videos"):
        os.makedirs("/app/output_dir/output/videos", exist_ok=True)
    output_path = os.path.join("/app/output_dir/output/videos", output_path)
    output_path = get_non_existing_path(output_path.replace(".mp4", ".fps.mp4"))
    
    frames = [resize_pil_image(frames[i]) for i in range(len(frames))]
    for findex in range(len(frames)):
        frames[findex] = to_tensors_transform(frames[findex])
        if not should_upscale:
            frames[findex] = frames[findex].unsqueeze(0)
        
    if should_upscale:
        call_callback("Upscaling video")
        frames = utils.upscale(cogvideo_model.upscaler_model, torch.stack(frames).to('cuda'), 'cuda', output_device="cpu")
        frames = [to_tensors_transform(resize_pil_image(to_pil_transform(frames[i].cpu()), True)).unsqueeze(0) for i in range(frames.size(0))]

    call_callback("Increasing video FPS")
    frames = rife_inference_with_latents(cogvideo_model.interpolation_model, torch.stack(frames))
    frames = rife_inference_with_latents(cogvideo_model.interpolation_model, torch.stack(frames))
    frames = [to_pil_transform(f[0]) for f in frames]

    call_callback("Saving video")
    export_to_video(frames, output_path, fps=8)


def generate_video(
    prompt: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    quant: bool = False,
    should_upscale: bool = False,
    should_use_pyramid: bool = False,
    strength: float = 0.8,
):
    output_path = output_path.replace(".mp4", "")
    

    image = None
    video = None
    vae_encode_decode = False
    
    if generate_type == 'v2vae':
        generate_type = 'v2v'
        vae_encode_decode = True
    
    cogvideo_model.load_models(
        use5b_model=True, 
        generate_type=generate_type, 
        use_pyramid=False,
        use_sageatt=generate_type == "i2v", 
        use_gguf=False
    )

    if num_videos_per_prompt > 1:
        prompt = [prompt] * num_videos_per_prompt
        generator = []
        for index in range(num_videos_per_prompt):
            generator.append(torch.Generator(device="cuda").manual_seed(seed + index))
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe_args = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_videos_per_prompt": 1, 
        "num_frames": 49,
        "use_dynamic_cfg": True,
    }
    
    if generate_type == "i2v":
        image = load_image_with_pil(image_or_video_path.strip())
        pipe_args["image"] = image
    elif generate_type != "t2v":
        video = load_video(image_or_video_path.strip())
        video_frames = len(video)
        max_frames = 49
        if video_frames > max_frames:
            video = video[:max_frames]
        pipe_args["strength"] = strength
        pipe_args["video"] = video

    cogvideo_model.pipe.progress_bar = lambda total: TqdmUpTo(total=total)


    call_callback("Generating video")
    
    if vae_encode_decode:
        vae = cogvideo_model.pipe.vae
        video = [to_tensors_transform(frame) for frame in video]
        video = torch.stack(video).to('cuda').permute(1, 0, 2, 3).unsqueeze(0).to(vae.dtype)
        with torch.no_grad():
            latents = vae.encode(video)[0].sample()
            video_generate = vae.decode(latents).sample
        tensor = video_generate.to(dtype=torch.float32)
        video_generate = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        video_generate = np.clip(video_generate, 0, 1) * 255
        video_generate = video_generate.astype(np.uint8)
        video_generate = [[Image.fromarray(video_generate[i]) for i in range(video_generate.shape[0])]]
    else:
        video_generate = cogvideo_model.pipe(**pipe_args).frames

    if len(video_generate) < 1:
        call_callback("No video generated.")
        return
    
    if num_videos_per_prompt < 2:
        save_video(video_generate[0], f'{output_path}_seed_{seed}_steps{num_inference_steps}.mp4', should_upscale)
    else:
        for i in range(len(video_generate)):
            save_video(video_generate[i], f"_seed_{seed}_steps{num_inference_steps}.{i}.mp4", should_upscale)


def iterate_prompts(prompt_path):
    from pipelines.cogvideo.prompt_parser import PromptStore
    store = PromptStore(prompt_path)
    while True:
        if SHOULD_STOP:
            raise StopException("Stopped by user.")
        added, removed = store.load()
        print(f"Added {added} prompts and removed {removed} prompts.")

        if len(store.prompts) == 0:
            raise StopException("No prompts found.")

        prompt = store.prompts[0]
        prompt.run_count += 1
        prompt_data = prompt.to_dict()
        if prompt_data["seed_use"] == -1:
            prompt_data["seed_use"] = random.randint(0, 100000)
        first_frame_pos = store.find_display_position_index()
        call_callback(f"Processing prompt {first_frame_pos + 1} of {len(store.prompts)}")

        yield prompt_data
        

def process_cogvideo_task_generate(task: dict) -> dict:
    try:
        prompt = task.get("prompt", None)
        if prompt is None:
            raise ValueError("The prompt should not be empty.")
        if len(prompt.strip()) < 1:
            raise ValueError("The prompt should not be empty.")
        lora_path = task.get("lora_path", None) # optional
        lora_rank = task.get("lora_rank", 128) # optional
        output_path = "/app/output_dir/output.mp4"
        generate_type = task.get("generate_type", "i2v")
        if generate_type not in ["t2v", "i2v", "v2v", "v2vae"]:
            raise ValueError("The generate type should be 't2v', 'i2v', 'v2v' or 'v2vae'.")
        image_or_video_path = task.get("image_or_video_path", "")
        if len(image_or_video_path.strip()) < 1 and  generate_type != "t2v":
            raise ValueError("The image or video path should not be empty.")
        num_inference_steps = task.get("num_inference_steps", 50)
        guidance_scale = task.get("guidance_scale", 6.0)
        num_videos_per_prompt = task.get("num_videos_per_prompt", 1)
        seed = task.get("seed", 42)
        quant = task.get("quant", False)
        loop = task.get("loop", False)
        should_upscale = task.get("should_upscale", False)
        use_pyramid = task.get("use_pyramid", False)
        strength = task.get("strength", 0.8)
            
        generate_video(
            prompt=prompt,
            lora_path=lora_path,
            lora_rank=lora_rank,
            output_path=output_path,
            image_or_video_path=image_or_video_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generate_type=generate_type,
            seed=seed,
            quant=quant,
            should_upscale=should_upscale,
            should_use_pyramid=use_pyramid,
            strength=strength,
        )
        
        return {
            "success": True,
        }
    except Exception as ex:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(ex)
        }
        
        
def process_prompts_from_file(prompts_path: str):
    dtype = torch.bfloat16
    file_index = 0
    output_path = 'output.mp4'
    saved_outpath = output_path
    args_lora_path = ''
    args_lora_rank = 128
    for prompt in iterate_prompts(prompts_path):
        args_output_path = saved_outpath
        seed = prompt["seed_use"]
        args_output_path = f'{args_output_path.split(".")[0]}_{file_index}.mp4'
        while os.path.exists(args_output_path):
            file_index += 1
            args_output_path = f'{args_output_path.split(".")[0]}.mp4'

        generate_video(
            prompt=prompt["prompt"],
            lora_path=args_lora_path,
            lora_rank=args_lora_rank,
            output_path=args_output_path,
            image_or_video_path=prompt["image"],
            num_inference_steps=prompt["steps"],
            guidance_scale=float(prompt["cfg"]),
            num_videos_per_prompt=prompt["num_videos_per_prompt"],
            dtype=dtype,
            generate_type=prompt["generate_type"],
            seed=seed,
            quant=prompt["quant"] in ['yes', 'true', '1'],
            # loop=(prompt["loop"].lower()  in ['yes', 'true', '1'] if isinstance(prompt["loop"], (str,)) else prompt["loop"]),
            should_upscale=prompt["should_upscale"],
            should_use_pyramid=prompt["use_pyramid"],
            strength=prompt["strength"],
        )

        if prompt.get("stoponthis", "") in ['yes', 'true', '1']:
            print("Only this prompt was generated due config.stoponthis in prompt.txt")
            break
    return {
        "success": True,
    }
    

def train_cogvideo_lora(train_filepath: str):
    print("Training LoRA model")
    train_lora_model(train_filepath)
    return {
        "success": False,
    }


def process_cogvideo_task(task: dict, callback = None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback

    SHOULD_STOP = False

    try:
        if 'prompt' in task:
            call_callback("Generating video from a prompt passed as parameter")
            return process_cogvideo_task_generate(task)
        if 'prompts_path' in task:
            call_callback("Iterating over a file and parsing prompts to generate videos")
            return process_prompts_from_file(task['prompts_path'])
        if 'train_file' in task:
            call_callback("Training lora model")
            return train_cogvideo_lora(task['train_file'])
        return {
            "success": False,
            "error": "Cogvideo: Invalid task",
        }
        SHOULD_STOP = False
    except StopException as ex:
        SHOULD_STOP = False
        print("Task stopped by the user.")
        return {
            "success": False,
            "error": str(ex)
        }
    except Exception as ex:
        SHOULD_STOP = False
        print(str(ex))
        traceback.print_exc()
        return {
            "success": False,
            "error": str(ex)
        }


def cancel_cogvideo_task():
    global SHOULD_STOP
    print("Cancelling CogVideo tasks if any...") 
    SHOULD_STOP = True
    return {
        "success": True,
    }
