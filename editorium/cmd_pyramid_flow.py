import urllib
import json
import click
import os

from .help_formater import call_command
from .request_wrapper import wait_task_completion, post_json_request, delete_request, cancel_task_request

from .common.path_utils import get_output_path
from .docker_management import full_path

   

@click.group(help="Manages Pyramid Flow")
def pyramid_group():
    pass


@pyramid_group.command(help='Generates video from text')
@click.option('--prompt', type=str, required=True, help="The prompt to generate the video")
@click.option('--seed', type=int, required=False, default=-1, help="The seed to generate the video")
def text2video(prompt, seed):
    parameters = {
        "prompt": prompt,
        "seed_use": seed,
        "generate_type": "t2v",
    }
    payload = {
        "task_type": "pyramid_flow",
        "parameters": parameters
    }

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
        

@pyramid_group.command(help='Generates video from image')
@click.option('--prompt', type=str, required=True, help="The prompt to generate the video")
@click.option('--image', type=str, required=True, help="The image to generate the video")
def image2video(prompt, image):
    parameters = {
        "prompt": prompt,
        "generate_type": "i2v",
        'image': get_output_path(image)
    }
    payload = {
        "task_type": "pyramid_flow",
        "parameters": parameters
    }

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

        
@pyramid_group.command(help='Processes a prompts file line by line and generate videos following the instructions')
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
        "task_type": "pyramid_flow",
        "parameters": parameters
    }

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


@pyramid_group.command(help='Cancels a task')
@click.option('--task-id', type=str, required=True, help="The task id")
def cancel_task(task_id):
    cancel_task_request("http://localhost:5000", task_id)
            

def register(main):
    @main.command(name='pyramid-flow', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Generate videos from text or image")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def pyramid_cmd(args):
        call_command(pyramid_cmd.name, pyramid_group, args)
