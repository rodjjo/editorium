import urllib
import json
import click
import os

from .help_formater import call_command
from .request_wrapper import wait_task_completion, post_json_request, delete_request, cancel_task_request
from .common.path_utils import get_output_path
from .docker_management import full_path


@click.group(help="Manages CogVideoX")
def cog_group():
    pass


@cog_group.command(help='Generates video from text')
@click.option('--prompt', type=str, required=True, help="The prompt to generate the video")
@click.option('--lora-path', type=str, default=None, help="The path to the LoRA model")
@click.option('--lora-rank', type=int, default=128, help="The rank of the LoRA model")
@click.option('--output-name', type=str, default="output.mp4", help="The name to the output video (ex.: output.mp4)")
@click.option('--num-inference-steps', type=int, default=50, help="The number of inference steps")
@click.option('--guidance-scale', type=float, default=6.0, help="The guidance scale")
@click.option('--num-videos-per-prompt', type=int, default=1, help="The number of videos per prompt")
@click.option('--seed', type=int, default=42, help="The seed")
@click.option('--quant', is_flag=True, help="Quantize the video")
@click.option('--loop', is_flag=True, help="Loop the video")
@click.option('--should-upscale', is_flag=True, help="Upscale the video")
@click.option('--should-use-pyramid', is_flag=True, help="Use pyramid")
def text2video(prompt, lora_path, lora_rank, output_name, num_inference_steps, guidance_scale, num_videos_per_prompt, seed, quant, loop, should_upscale, should_use_pyramid):
    lora_path = get_output_path(lora_path)
    parameters = {
        "prompt": prompt,
        "generate_type": "t2v",
        "lora_path": lora_path,
        "lora_rank": lora_rank,
        "output_name": output_name,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
        "seed": seed,
        "quant": quant,
        "loop": loop,
        "should_upscale": should_upscale,
        "should_use_pyramid": should_use_pyramid,
    }
    payload = {
        "task_type": "cogvideo",
        "parameters": parameters
    }
    # makes a post request to http://localhost:5000/tasks using python urllib library with the payload
    try:
        data = post_json_request("http://localhost:5000/tasks", payload)  
        if data.get('task_id', None):
            print(f'Task {data["task_id"]} created')
            # wait_task_completion(data['task_id'])
        else:
            print(data)
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)


@cog_group.command(help='Generates video from image')
@click.option('--prompt', type=str, required=True, help="The prompt to generate the video")
@click.option('--image-path', type=str, required=True, help="The path to the image")
@click.option('--lora-path', type=str, default=None, help="The path to the LoRA model")
@click.option('--lora-rank', type=int, default=128, help="The rank of the LoRA model")
@click.option('--output-name', type=str, default="output.mp4", help="The name to the output video (ex.: output.mp4)")
@click.option('--num-inference-steps', type=int, default=50, help="The number of inference steps")
@click.option('--guidance-scale', type=float, default=6.0, help="The guidance scale")
@click.option('--num-videos-per-prompt', type=int, default=1, help="The number of videos per prompt")
@click.option('--seed', type=int, default=42, help="The seed")
@click.option('--quant', is_flag=True, help="Quantize the video")
@click.option('--loop', is_flag=True, help="Loop the video")
@click.option('--should-upscale', is_flag=True, help="Upscale the video")
@click.option('--should-use-pyramid', is_flag=True, help="Use pyramid")
def image2video(prompt, image_path, lora_path, lora_rank, output_name, num_inference_steps, guidance_scale, num_videos_per_prompt, seed, quant, loop, should_upscale, should_use_pyramid):
    image_path = get_output_path(image_path)
    lora_path = get_output_path(lora_path)
    parameters = {
        "prompt": prompt,
        "image_or_video_path": image_path,
        "generate_type": "i2v",
        "lora_path": lora_path,
        "lora_rank": lora_rank,
        "output_name": output_name,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
        "seed": seed,
        "quant": quant,
        "loop": loop,
        "should_upscale": should_upscale,
        "should_use_pyramid": should_use_pyramid,
    }
    payload = {
        "task_type": "cogvideo",
        "parameters": parameters
    }
    # makes a post request to http://localhost:5000/tasks using python urllib library with the payload
    try:
        data = post_json_request("http://localhost:5000/tasks", payload)  
        if data.get('task_id', None):
            print(f'Task {data["id"]} created')
            # wait_task_completion(data['task_id'])
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)


@cog_group.command(help='Generates video from video')
@click.option('--prompt', type=str, required=True, help="The prompt to generate the video")
@click.option('--video-path', type=str, required=True, help="The path to the video")
@click.option('--lora-path', type=str, default=None, help="The path to the LoRA model")
@click.option('--lora-rank', type=int, default=128, help="The rank of the LoRA model")
@click.option('--output-name', type=str, default="output.mp4", help="The name to the output video (ex.: output.mp4)")
@click.option('--num-inference-steps', type=int, default=50, help="The number of inference steps")
@click.option('--guidance-scale', type=float, default=6.0, help="The guidance scale")
@click.option('--num-videos-per-prompt', type=int, default=1, help="The number of videos per prompt")
@click.option('--seed', type=int, default=42, help="The seed")
@click.option('--quant', is_flag=True, help="Quantize the video")
@click.option('--loop', is_flag=True, help="Loop the video")
@click.option('--should-upscale', is_flag=True, help="Upscale the video")
@click.option('--should-use-pyramid', is_flag=True, help="Use pyramid")
@click.option('--strength', type=float, default=0.8, help="The strength")
def video2video(prompt, video_path, lora_path, lora_rank, output_name, num_inference_steps, guidance_scale, num_videos_per_prompt, seed, quant, loop, should_upscale, should_use_pyramid, strength):
    video_path = get_output_path(video_path)
    lora_path = get_output_path(lora_path)
    parameters = {
        "prompt": prompt,
        "image_or_video_path": video_path,
        "generate_type": "v2v",
        "lora_path": lora_path,
        "lora_rank": lora_rank,
        "output_name": output_name,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
        "seed": seed,
        "quant": quant,
        "loop": loop,
        "should_upscale": should_upscale,
        "should_use_pyramid": should_use_pyramid,
        "strength": strength
    }
    payload = {
        "task_type": "cogvideo",
        "parameters": parameters
    }
    # makes a post request to http://localhost:5000/tasks using python urllib library with the payload
    try:
        data = post_json_request("http://localhost:5000/tasks", payload)  
        if data.get('task_id', None):
            print(f'Task {data["task_id"]} created')
            # wait_task_completion(data['task_id'])
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)
        

@cog_group.command(help='Processes a prompts file line by line and generate videos following the instructions')
@click.option('--prompts-path', type=str, required=True, help="The path to the prompts file")
def generate_from_file(prompts_path):
    prompts_path = full_path(prompts_path)
    if os.path.exists(prompts_path) is False:
        print(f"File {prompts_path} not found")
        return

    with open(prompts_path, 'r') as f:
        file_content = f.read()
    
    parameters = {
        "prompts_data": file_content
    }
    payload = {
        "task_type": "cogvideo",
        "parameters": parameters
    }
    # makes a post request to http://localhost:5000/tasks using python urllib library with the payload
    try:
        data = post_json_request("http://localhost:5000/tasks", payload)  
        if data.get('task_id', None):
            print(f'Task {data["task_id"]} created')
            # wait_task_completion(data['task_id'])
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)
        

@cog_group.command(help='Fine tune CogvideoX-5B lora model')
#   --train-file /app/data/train.json
@click.option('--train-file', type=str, required=True, help="The path to the train file")
def fine_tune(train_file):
    parameters = {
        "train_file": train_file
    }
    payload = {
        "task_type": "cogvideo_lora",
        "parameters": parameters
    }
    # makes a post request to http://localhost:5000/tasks using python urllib library with the payload
    try:
        data = post_json_request("http://localhost:5000/tasks", payload)  
        if data.get('task_id', None):
            print(f'Task {data["task_id"]} created')
            # wait_task_completion(data['task_id'])
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)


@cog_group.command(help='Cancels a task')
@click.option('--task-id', type=str, required=True, help="The task id")
def cancel_task(task_id):
    cancel_task_request("http://localhost:5000", task_id)
            

def register(main):
    @main.command(name='cogvideo', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Generate videos from text, image or other video")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def cog_cmd(args):
        call_command(cog_cmd.name, cog_group, args)
