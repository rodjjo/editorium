import click
import os
import re
import urllib

from .docker_management import full_path
from .help_formater import call_command
from .request_wrapper import post_json_request, get_request
from .docker_management import full_path


@click.group()
def workflow_group():
    pass


def read_worflow_file(include_dir: str, path: str, already_included: set, replace_input: dict):
    if include_dir == '':
        path = full_path(path)
        include_dir = os.path.dirname(path)
    else:
        path = full_path(os.path.join(include_dir, path))
    if os.path.exists(path) is False:
        raise Exception(f"File {path} not found")
    if path in already_included:
        raise Exception(f"File {path} already included")
    already_included.add(path)
    parsed_lines = []
    capture_inputs1 = re.compile('#input=([^ $]+)[ $]*')
    capture_inputs2 = re.compile('#input\\.([^=]+)=([^ $]+)[ $]*')
    capture_path = re.compile('#path=([^$]+)$')
    with open(path, 'r') as f:
        file_content = f.readlines()
        if replace_input:
            for key in replace_input:
                for index, line in enumerate(file_content):
                    if f"task://<{key}>" in line:
                        file_content[index] = line.replace(f"task://<{key}>", f'task://{replace_input[key]}')
                    if f"from://<{key}>" in line:
                        file_content[index] = line.replace(f"from://<{key}>", f'from://{replace_input[key]}')

        for line in file_content:
            line = line.strip()  # #include #input=bla #path=bla
            if line.startswith("#comment"):
                continue
            if line.startswith("#include "):
                inputs1 = re.search(capture_inputs1, line)
                inputs2 = re.findall(capture_inputs2, line)
                parsed_inputs = {}
                if inputs1:
                    parsed_inputs["input"] = inputs1.group(1).strip()
                if inputs2:
                    for key, value in inputs2:
                        parsed_inputs[f'input.{key}'] = value
                parsed_path = re.search(capture_path, line)
                if parsed_path:
                    path = parsed_path.group(1).strip()
                else:
                    raise Exception(f"Invalid include line: {line} it does not have #path=value")
                parsed_lines += read_worflow_file(include_dir, path, already_included, parsed_inputs)
            else:
                parsed_lines.append(line)
    return parsed_lines
                
        
@workflow_group.command(help='Processes a workflow file line by line and generate photo and videos following the instructions')
@click.option('--path', type=str, required=True, help="The path to the workflow file")
def run(path):
    parameters = {
        "workflow": read_worflow_file('', path, set(), {})
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
