import click
import os

import urllib

from .docker_management import full_path
from .help_formater import call_command
from .request_wrapper import post_json_request, get_request
from .docker_management import full_path


@click.group()
def workflow_group():
    pass

        
@workflow_group.command(help='Processes a workflow file line by line and generate photo and videos following the instructions')
@click.option('--path', type=str, required=True, help="The path to the workflow file")
def run(path):
    path = full_path(path)
    if os.path.exists(path) is False:
        print(f"File {path} not found")
        return
    with open(path, 'r') as f:
        file_content = f.readlines()
    parameters = {
        "workflow": [l.strip() for l in file_content]
    }
    payload = {
        "task_type": "workflow",
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


@workflow_group.command(help='Shows a list of tasks that can be executed by a workflow')
def show_workflow_tasks():
    try:
        data = get_request("http://localhost:5000/workflow-tasks")
        for task in data:
            print(f'{task["task_type"]}: {task["description"]}\n\n')
        else:
            print(data)
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)

def register(main):
    @main.command(name='workflow', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Manages a workflow that allow complex operations to be executed in a sequence")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def desktop_cmd(args):
        call_command(desktop_cmd.name, workflow_group, args)
