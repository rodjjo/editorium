from typing import List

from pipelines.common import utils
from pipelines.common.rife_model import rife_inference_with_latents
from pipelines.common.save_video import save_video, to_tensors_transform
from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException

from pipelines.pyramid_flow.managed_model import pyramid_model


class PyramidTaskParameters:
    prompt: str = ''
    generate_type: str = 't2v'
    seed: int = 42
    num_inference_steps: List[int] = [20, 20, 20]
    video_num_inference_steps: List[int] = [10, 10, 10]
    height: int = 768
    width: int = 1280
    temp: int = 16                      # temp=16: 5s, temp=31: 10s
    guidance_scale : float = 9.0,       # The guidance for the first frame, set it to 7 for 384p variant
    video_guidance_scale: float = 5.0   # The guidance for the other video latent
    use5b_model: bool = True
    input_image: str = None

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
            'seed': self.seed,
            'use5b_model': self.use5b_model,
            'input_image': self.input_image,
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
        params.seed = data.get('seed', 42)
        params.use5b_model = data.get('use5b_model', True)
        params.input_image = data.get('input_image', None)
        return params


def generate_video(task: PyramidTaskParameters) -> dict:
    if task.generate_type not in ['t2v', 'i2v']:
        raise ValueError('Invalid generate_type')

    pyramid_model.load_models(
        generate_type=task.generate_type,
        use5b_model=task.use5b_model,
    )
    pipe_args = {
        'prompt': task.prompt,
        'num_inference_steps': task.num_inference_steps,
        'temp': task.temp,
        'video_guidance_scale': task.video_guidance_scale,
        'save_memory': True,
        'cpu_offloading': True,
    }
    
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
        image = utils.load_image_rgb(task.input_image)
        pipe_args = {
            **pipe_args,
            'input_image': image,
        }
        video_generate = pyramid_model.pipeline.generate_i2v(
            **pipe_args
        )
    
    save_video(
        video_generate, 
        f"pyramid_seed_{task.seed}_steps{task.num_inference_steps[0]}.mp4", 
        pyramid_model.upscaler_model,
        pyramid_model.interpolation_model,
        False
    )
    
    return {
        'status': 'success',
        'message': 'Video generation completed successfully'
    }


def process_pyramid_task(parameters: dict, callback = None) -> dict:
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
    pass