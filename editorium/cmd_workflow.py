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


def read_worflow_file(include_dir: str, path: str, already_included: set, replace_input: dict, sufix: str):
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
    capture_inputs1 = re.compile('#input=([^#]+)')
    capture_inputs2 = re.compile('#input\\.([^=]+)=([^#]+)')
    capture_path = re.compile('.*#path=([^$#]+).*')
    capture_sufix = re.compile('.*#suffix=([0-9]+).*')
    
    with open(path, 'r') as f:
        file_content = f.readlines()
        
        if sufix:
            for index, line in enumerate(file_content):
                if line.startswith("#name="):
                    file_content[index] = f'{line.strip()}-{sufix}'
                    continue
                if 'from://<' in line or 'task://<' in line:
                    continue
                if 'from://' in line or 'task://' in line:
                    file_content[index] = f'{line.strip()}-{sufix}'

        if replace_input:
            print("Replacing inputs of file ", path, " with ", replace_input)
            for key in replace_input:
                for index, line in enumerate(file_content):
                    if f"task://<{key}>" in line:
                        file_content[index] = line.replace(f"task://<{key}>", f'task://{replace_input[key]}')
                    elif f"from://<{key}>" in line:
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
                    for input in inputs2:
                        parsed_inputs[f'input.{input[0].strip()}'] = input[1].strip()
                print(parsed_inputs)
                parsed_path = re.match(capture_path, line)
                parsed_sufix = re.match(capture_sufix, line)
                if not parsed_sufix:
                    raise Exception(f"Invalid include line: {line} it does not have #suffix=value")
                if not parsed_path:
                    raise Exception(f"Invalid include line: {line} it does not have #path=value")
                
                path = parsed_path.group(1).strip()
                sufix = parsed_sufix.group(1).strip()
                
                parsed_lines += read_worflow_file(include_dir, path, already_included, parsed_inputs, sufix)
            else:
                parsed_lines.append(line)
    return parsed_lines
                
        
@workflow_group.command(help='Processes a workflow file line by line and generate photo and videos following the instructions')
@click.option('--path', type=str, required=True, help="The path to the workflow file")
def run(path):
    parameters = {
        "workflow": read_worflow_file('', path, set(), {}, '')
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
