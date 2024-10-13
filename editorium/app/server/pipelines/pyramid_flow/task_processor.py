from typing import List

import os
import torch
from tqdm import tqdm

from pipelines.common import utils
from pipelines.common.rife_model import rife_inference_with_latents
from pipelines.common.save_video import save_video, to_tensors_transform
from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.pyramid_flow.managed_model import pyramid_model


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

def progress_callback(position, total):
    if SHOULD_STOP:
        raise StopException("Stopped by user.")
    if PROGRESS_CALLBACK is not None and total is not None and total > 0:
        PROGRESS_CALLBACK(CURRENT_TITLE, position / total)



class PyramidTaskParameters:
    prompt: str = ''
    generate_type: str = 't2v'
    seed_use: int = 42
    num_inference_steps: List[int] = [20, 20, 20]
    video_num_inference_steps: List[int] = [10, 10, 10]
    height: int = 768
    width: int = 1280
    temp: int = 31                      # temp=16: 5s, temp=31: 10s
    guidance_scale : float = 9.0,       # The guidance for the first frame, set it to 7 for 384p variant
    video_guidance_scale: float = 5.0   # The guidance for the other video latent
    use768p_model: bool = True
    image: str = None
    stoponthis: bool = False

    def __init__(self):
        pass
    
    def to_dict(self) -> dict:
        return {
            'generate_type': self.generate_type,
            'prompt': self.prompt,
            'num_inference_steps': self.num_inference_steps,
            'video_num_inference_steps': self.video_num_inference_steps,
            'height': self.height,
            'width': self.width,
            'temp': self.temp,
            'guidance_scale': self.guidance_scale,
            'video_guidance_scale': self.video_guidance_scale,
            'seed_use': self.seed,
            'use768p_model': self.use768p_model,
            'image': self.image,
            'stoponthis': self.stoponthis,
        }
    
    @staticmethod
    def from_dict(data: dict):
        params = PyramidTaskParameters()
        params.generate_type = data.get('generate_type', 't2v')
        params.prompt = data.get('prompt', '')
        params.num_inference_steps = data.get('num_inference_steps', [20, 20, 20])
        params.video_num_inference_steps = data.get('video_num_inference_steps', [10, 10, 10])
        params.height = data.get('height', 768)
        params.width = data.get('width', 1280)
        params.temp = data.get('temp', 16)
        params.guidance_scale = data.get('guidance_scale', 9.0)
        params.video_guidance_scale = data.get('video_guidance_scale', 5.0)
        params.seed_use = data.get('seed', 42)
        params.use768p_model = data.get('use768p_model', True)
        params.image = data.get('image', None)
        params.stoponthis = data.get('stoponthis', False)
        return params


def generate_video(task: PyramidTaskParameters) -> dict:
    if task.generate_type not in ['t2v', 'i2v']:
        raise ValueError('Invalid generate_type')
   
    pyramid_model.load_models(
        generate_type=task.generate_type,
        use768p_model=task.use768p_model,
    )
    pipe_args = {
        'prompt': task.prompt,
        'num_inference_steps': task.num_inference_steps,
        'temp': task.temp,
        'video_guidance_scale': task.video_guidance_scale,
        'save_memory': True,
        'cpu_offloading': True,
        'callback': progress_callback,
    }
    
    if task.seed_use in (None, -1):
        seed = torch.randint(0, 1000000, (1,)).item()
    else:
        seed = task.seed_use
        
    pipe_args['generator'] = torch.Generator(device="cuda").manual_seed(seed)
        
    if task.generate_type == 't2v':
        pipe_args = {
            **pipe_args,
            'video_num_inference_steps': task.video_num_inference_steps,
            'height': task.height,
            'width': task.width,
            'guidance_scale': task.guidance_scale,
        }
        video_generate = pyramid_model.pipeline.generate(
            **pipe_args
        )
    else:
        image = utils.load_image_rgb(task.image)
        expected_width = task.use768p_model and 1280 or 640
        expected_height = task.use768p_model and 768 or 384
        if image.width != expected_width or image.height != expected_height:
            image = image.resize((expected_width, expected_height))
        pipe_args = {
            **pipe_args,
            'input_image': image,
        }
        video_generate = pyramid_model.pipeline.generate_i2v(
            **pipe_args
        )
    
    save_video(
        video_generate, 
        f"pyramid_seed_{seed}_steps{task.num_inference_steps[0]}.mp4", 
        fps=24
    )
    
    return {
        'status': 'success',
        'message': 'Video generation completed successfully'
    }

def process_prompts_from_file(prompts_data: str):
    dtype = torch.bfloat16
    file_index = 0
    output_path = 'output.mp4'
    saved_outpath = output_path

    
    prompts_dir = '/app/output_dir/prompts'
    os.makedirs(prompts_dir, exist_ok=True)
    prompts_path = os.path.join(prompts_dir, 'pyramid_prompts.txt')
    
    with open(prompts_path, "w") as f:
        f.write(prompts_data)
    
    for prompt, pos, count in iterate_prompts(prompts_path, 'pyramid'):
        if SHOULD_STOP:
            print("Stopped by user.")
            break

        print(f"Processing prompt {pos + 1}/{count}")

        args_output_path = saved_outpath
        
        prompt = PyramidTaskParameters.from_dict(prompt)
        
        args_output_path = f'{args_output_path.split(".")[0]}_pyramid_{file_index}.mp4'
        while os.path.exists(args_output_path):
            file_index += 1
            args_output_path = f'{args_output_path.split(".")[0]}.mp4'

        generate_video(prompt)

        if prompt.stoponthis in ['yes', 'true', '1', True]:
            print("Only this prompt was generated due config.stoponthis in prompt.txt")
            break
    return {
        "success": True,
    }
    

def process_pyramid_task(parameters: dict, callback = None) -> dict:
    global SHOULD_STOP
    SHOULD_STOP = False
    if 'prompts_data' in parameters:
        return process_prompts_from_file(parameters['prompts_data'])
    else:
        parameters = PyramidTaskParameters.from_dict(parameters)
        try:
            result = generate_video(parameters)
        except StopException as e:
            result = {
                'status': 'error',
                'message': str(e)
            }
        except Exception as e:
            # import required modules and print call stack
            import traceback
            traceback.print_exc()
            print(e)
            result = {
                'status': 'error',
                'message': str(e)
            }


def cancel_pyramid_task() -> dict:
    global SHOULD_STOP
    SHOULD_STOP = True
    return {
        'status': 'success',
        'message': 'Task scheduled to stop successfully'
    }