
import gc
import traceback
import os
import sys
import random
import argparse
from typing import Literal, List
import torch
from diffusers import (
    AutoencoderKLCogVideoX, 
    CogVideoXTransformer3DModel, 
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video, load_video
from transformers import T5EncoderModel
from torchvision import transforms
from PIL import Image
import pipelines.cogvideo.utils as utils
from pipelines.cogvideo.rife_model import load_rife_model, rife_inference_with_latents
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
from torchao.quantization.utils import recommended_inductor_config_setter

from PIL import Image

import subprocess
from tqdm import tqdm
import numpy as np

# from torchao.float8.inference import ActivationCasting, QuantConfig, quantize_to_float8

from pipelines.cogvideo.cogvideox_transformer_att import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelAtt
from pipelines.cogvideo.cogvideox_transformer_pab import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelPAB
from pipelines.cogvideo.core.pab_mgr import set_pab_manager, CogVideoXPABConfig
from pipelines.cogvideo.load_gguf import load_gguf_transformer
from pipelines.cogvideo.cogvideox_lora_trainer import train_lora_model

# os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

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



class StopException(Exception):
    pass

# recommended_inductor_config_setter()

if not os.path.exists("/home/editorium/models/upscalers"):
    os.makedirs("/home/editorium/models/upscalers", exist_ok=True)
    
if not os.path.exists("/home/editorium/models/interpolations"):
    os.makedirs("/home/editorium/models/interpolations", exist_ok=True)

upscale_model = None
frame_interpolation_model = None


def quantize_model(part, quantization_scheme="fp8"):
    # if quantization_scheme == "int8":
    quantize_(part, int8_weight_only())
    #elif quantization_scheme == "fp8":
    # quantize_to_float8(part, QuantConfig(ActivationCasting.DYNAMIC))
    return part


def quantize_pipe(model_path, model_class, dtype, quantization_scheme="fp8", use_attn=False, use_gguf=False):
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    if quantization_scheme:
        text_encoder = quantize_model(part=text_encoder, quantization_scheme=quantization_scheme)
    
    if use_attn:
        transform_class = CogVideoXTransformer3DModelAtt
    else:
        transform_class = CogVideoXTransformer3DModel
        
    if use_gguf:
        #  CogVideoX_5b_I2V_GGUF_Q4_0.safetensors    
        transformer = transform_class.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
        gguf_path = "/home/editorium/models/videos/CogVideoX_5b_I2V_GGUF_Q4_0.safetensors"
        if os.path.exists(gguf_path):
            load_gguf_transformer(transformer, gguf_path)
    else: 
        transformer = transform_class.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
        
    if quantization_scheme:
        transformer = quantize_model(part=transformer, quantization_scheme=quantization_scheme)
        
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    if quantization_scheme:
        vae = quantize_model(part=vae, quantization_scheme=quantization_scheme)
        
    #if quantization_scheme:
    #    transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)
    #if not quantization_scheme:
    #    transformer = transformer.to(memory_format=torch.channels_last)
    #    transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)

    gc.collect()
    torch.cuda.empty_cache()
    pipe = model_class.from_pretrained(
        model_path.strip(), 
        torch_dtype=dtype,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder
    )
    gc.collect()
    torch.cuda.empty_cache()
    if not quantization_scheme:
        pipe.to(dtype=torch.float16)
    return pipe


def load_image_with_pil(image_path: str):
    """
    Load an image from the given path using PIL.

    Parameters:
    - image_path (str): The path of the image to be loaded.

    Returns:
    - image (PIL.Image): The loaded image.
    """
    
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

pipe = None
current_type = None
current_model = None
current_looping = False
current_pyramid = False

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
        frames = utils.upscale(upscale_model, torch.stack(frames).to('cuda'), 'cuda', output_device="cpu")
        frames = [to_tensors_transform(resize_pil_image(to_pil_transform(frames[i].cpu()), True)).unsqueeze(0) for i in range(frames.size(0))]

    call_callback("Increasing video FPS")
    frames = rife_inference_with_latents(frame_interpolation_model, torch.stack(frames))
    frames = rife_inference_with_latents(frame_interpolation_model, torch.stack(frames))
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
    loop: bool = False,
    should_upscale: bool = False,
    should_use_pyramid: bool = False,
    strength: float = 0.8,
):
    global pipe
    global current_type
    global current_looping
    global current_model
    global current_pyramid
    global upscale_model
    global frame_interpolation_model
    output_path = output_path.replace(".mp4", "")
    
    if upscale_model is None:
        upscale_model = utils.load_sd_upscale('/home/editorium/models/upscalers/RealESRGAN_x4.pth', 'cpu')
        
    if frame_interpolation_model is None:
        frame_interpolation_model = load_rife_model("/home/editorium/models/interpolations/model_rife")


    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None
    vae_encode_decode = False
    
    if current_type != generate_type:
        call_callback(f"Loading model to generate video. type = {generate_type}")
        pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        current_type = generate_type

    loop_enabled = loop is True or isinstance(loop, int)
    if current_looping != loop_enabled:
        call_callback(f"Loading model to generate video. loop = {loop}")
        pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        current_looping = loop_enabled
    
    if current_pyramid != should_use_pyramid:
        call_callback(f"Loading model to generate video. pyramid = {should_use_pyramid}")
        pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        current_pyramid = should_use_pyramid
    
    if generate_type == "i2v":
        model_path = "THUDM/CogVideoX-5b-I2V" 
    elif should_use_pyramid: 
        model_path =  "THUDM/CogVideoX-2b"
    elif generate_type == 'v2vae':
        generate_type = 'v2v'
        vae_encode_decode = True
        model_path =  "THUDM/CogVideoX-5b"
    else:
        model_path =  "THUDM/CogVideoX-5b"
    
        
    if current_model != model_path:
        call_callback(f"Loading model to generate video. model = {model_path}")
        pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        current_model = model_path
    

    if generate_type == "i2v":
        image = load_image_with_pil(image_or_video_path.strip())
    elif generate_type != "t2v":
        video = load_video(image_or_video_path.strip())
        video_frames = len(video)
        max_frames = 49
        if video_frames > max_frames:
            video = video[:max_frames]
    

    if pipe is None:
        if generate_type == "i2v":
            model_class = CogVideoXImageToVideoPipeline
        elif generate_type == "t2v":
            model_class = CogVideoXPipeline
        else:
            model_class = CogVideoXVideoToVideoPipeline
            
        if should_use_pyramid:
            config = CogVideoXPABConfig()
            set_pab_manager(config)
        
        use_gguf = False # generate_type == "i2v"
        if quant:
            call_callback("Quantizing model")
            pipe = quantize_pipe(model_path, model_class, dtype, "int8", use_attn=False, use_gguf=use_gguf)
            call_callback("Model quantized")
            generator = torch.Generator().manual_seed(seed)
            #pipe.vae.enable_slicing()
            #pipe.vae.enable_tiling()
            # print("Compiling model")
            # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)  
            # print("Model compiled")
            # pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = quantize_pipe(model_path, model_class, dtype, "", use_attn=generate_type=='i2v', use_gguf=use_gguf)
            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            # pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
        #if should_use_pyramid:
            #print("Enabling pyramid attention")
            #pipe.enable_pyramid_attention_broadcast(
            #    spatial_attn_skip_range=2,
            #    spatial_attn_timestep_range=[100, 850],
            #)

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        if vae_encode_decode:
            pipe = pipe.vae
        else:
            pipe.progress_bar = lambda total: TqdmUpTo(total=total)
            
            
        gc.collect()    
        torch.cuda.empty_cache()       
    

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
    # num_videos_per_prompt = 1
    if num_videos_per_prompt > 1:
        prompt = [prompt] * num_videos_per_prompt
        generator = []
        for index in range(num_videos_per_prompt):
            generator.append(torch.Generator(device="cuda").manual_seed(seed + index))
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    call_callback("Generating video")
    if generate_type == "i2v":
        add_args = {            
        }
        # if loop_enabled:
        # add_args['looping'] = loop
        video_generate = pipe(
            prompt=prompt,
            image=image,  # The path of the image to be used as the background of the video
            num_videos_per_prompt=1,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=True,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=generator,  # Set the seed for reproducibility
            **add_args,
        ).frames
    elif generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames
    else:
        if vae_encode_decode:
            vae = pipe
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
            video_generate = pipe(
                prompt=prompt,
                video=video,  # The path of the video to be used as the background of the video
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,  # Set the seed for reproducibility
                strength=strength,
            ).frames

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
            loop=loop,
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
            loop=(prompt["loop"].lower()  in ['yes', 'true', '1'] if isinstance(prompt["loop"], (str,)) else prompt["loop"]),
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
    cogvideo_release_resources()
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


def cogvideo_release_resources():
    global pipe
    global upscale_model
    global frame_interpolation_model
    pipe = None
    upscale_model = None
    frame_interpolation_model = None
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "success": True,
    }
