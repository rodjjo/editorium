import urllib
import json
import click
import os

from .help_formater import call_command
from .request_wrapper import wait_task_completion, post_json_request, delete_request, cancel_task_request


def full_path(path: str) -> str:
    if path.startswith('/'):
        # remove the first '/' if it exists
        path = path[1:]
    return os.path.join('/app/output_dir', path)
    

@click.group(help="Manages Pyramid Flow")
def pyramid_group():
    pass


@pyramid_group.command(help='Cancels a task')
@click.option('--task-id', type=str, required=True, help="The task id")
def cancel_task(task_id):
    cancel_task_request("http://localhost:5000", task_id)
            

def register(main):
    @main.command(name='pyramid', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Generate videos from text or image")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def pyramid_cmd(args):
        call_command(pyramid_cmd.name, pyramid_group, args)
