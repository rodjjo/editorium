import urllib
import json
import click
import os

from .help_formater import call_command
from .request_wrapper import wait_task_completion, post_json_request, delete_request, cancel_task_request

from .common.path_utils import get_output_path
from .docker_management import full_path

   
@click.group(help="Diverse utils commands")
def utils_group():
    pass


@utils_group.command(help='Convert a flux transformer model into diffusers format')
@click.option('--repo_id', type=str, required=True, help="The base repository id to get other components ")
@click.option('--unet_filename', type=str, required=True, help="The safetensor of the transformer to be converted")
def convert_flux_transformer(repo_id, unet_filename):
    parameters = {
        "command": "convert_flux_transformer",
        "repo_id": repo_id,
        "unet_filename": unet_filename,
    }
    payload = {
        "task_type": "utils",
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
        

@utils_group.command(help='Convert a flux model with a different transformer into diffusers format')
@click.option('--repo_id', type=str, required=True, help="The base repository id to get other components ")
@click.option('--unet_filename', type=str, required=True, help="The safetensor of the transformer to be converted")
def convert_flux_model(repo_id, unet_filename):
    parameters = {
        "command": "convert_flux_model",
        "repo_id": repo_id,
        "unet_filename": unet_filename,
    }
    payload = {
        "task_type": "utils",
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


def register(main):
    @main.command(name='utils', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Generate videos from text or image")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def utils_cmd(args):
        call_command(utils_cmd.name, utils_group, args)


