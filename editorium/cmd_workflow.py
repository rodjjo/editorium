import click
import os
import re
import urllib
from typing import List

from .docker_management import full_path
from .help_formater import call_command
from .request_wrapper import post_json_request, get_request
from .docker_management import full_path


@click.group()
def workflow_group():
    pass


def get_lines_from_file(path: str):
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()
        line_sum = ''
        include_started = False
        for line in lines:
            line = line.strip()
            if line.startswith('#include '):
                include_started = True
                
            if line.endswith('\\') and include_started:
                line = line[:-1]
                line_sum += line
            else: 
                line_sum += line
                include_started = False
                result.append(line_sum)
                line_sum = ''

        if line_sum:
            result.append(line_sum)
    return result


def read_worflow_file(include_dir: str, path: str, already_included: set, replace_input: dict, suffix: str):
    if include_dir == '':
        path = full_path(path)
        include_dir = os.path.dirname(path)
    else:
        path = full_path(os.path.join(include_dir, path))
    if os.path.exists(path) is False:
        raise Exception(f"File {path} not found")
    included_track = f'{path}-{suffix}'
    if included_track in already_included:
        raise Exception(f"File {path} already included")
    already_included.add(included_track)
    parsed_lines = []
    capture_inputs1 = re.compile(r'#input=([^#]+)')
    capture_inputs2 = re.compile(r'#input\\.([^=]+)=([^#]+)')
    capture_path = re.compile(r'.*#path=([^$#]+).*')
    capture_suffix = re.compile(r'.*#suffix=([0-9a-zA-Z_\-]+).*')
    
    file_content = get_lines_from_file(path)
        
    if suffix:
        replace_include_from = re.compile('<insert_task:([^<>]+)>')
        replace_suffix = re.compile(r'task://([^<>:]+)(:|$)')
        for index, line in enumerate(file_content):
            line = line.strip()
            if line.startswith("#name="):
                file_content[index] = f'{line.strip()}-{suffix}'
                continue
            file_content[index] = re.sub(replace_include_from, rf'<insert_task:\g<1>-{suffix}>', line)
            file_content[index] = re.sub(replace_suffix, rf'task://\g<1>-{suffix}\g<2>', line)    
            
    if replace_input:
        print("Replacing inputs of file ", path, " with ", replace_input)
        for key in replace_input:
            for index, line in enumerate(file_content):
                line = line.strip()
                if f"task://<{key}>" in line:
                    file_content[index] = line.replace(f"task://<{key}>", f'task://{replace_input[key]}')
                while f"<insert_task:<{key}>>" in line:
                    line = line.replace(f"<insert_task:<{key}>>", f'<insert_task:{replace_input[key]}>')
                    file_content[index] = line
                    
    for index, line in enumerate(file_content):
        line = line.strip()
        if "<insert_task:<" in line or "from://<" in line or "task://<" in line:
            print(f"[WARNING] Invalid include line: {line} it does not have #suffix=value")

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
            parsed_suffix = re.match(capture_suffix, line)
            if not parsed_suffix:
                raise Exception(f"Invalid include line: {line} it does not have #suffix=value")
            if not parsed_path:
                raise Exception(f"Invalid include line: {line} it does not have #path=value")
            
            path = parsed_path.group(1).strip()
            suffix = parsed_suffix.group(1).strip()
            
            parsed_lines += read_worflow_file(include_dir, path, already_included, parsed_inputs, suffix)
        else:
            parsed_lines.append(line)
    return parsed_lines


def extract_workflow_collections(include_dir: str, content: List[str], collections: dict) -> dict:
    in_task = False
    task_type = ''
    config_path = ''
    for line in content:
        line = line.strip()
        if line.startswith('#start'):
            in_task = True
            continue
        if line.startswith('#end'):
            if task_type == 'execute':
                if not config_path:
                    raise Exception(f"Invalid execute task configuration missing path: {line}")
                config_path_complete = full_path(os.path.join(include_dir, config_path))
                if os.path.exists(config_path_complete) is False:
                    raise Exception(f"Invalid execute task configuration path not found: {config_path_complete}")
                if config_path not in collections:
                    collections[config_path] = read_worflow_file(include_dir, config_path, set(), {}, '')
                    collections = extract_workflow_collections(os.path.dirname(config_path_complete), collections[config_path], collections)
            in_task = False
            task_type = ''
            config_path = ''
            continue
        if in_task:
            if line.startswith('#type='):
                task_type = line[6:].strip()
            elif line.startswith('#task_type='):
                task_type = line[11:].strip()
            elif line.startswith('#config.path='):
                config_path = line[13:].strip()
            if task_type and task_type != 'execute':
                config_path = ''            
                
    return collections
                
        
@workflow_group.command(help='Processes a workflow file line by line and generate photo and videos following the instructions')
@click.option('--path', type=str, required=True, help="The path to the workflow file")
def run(path):
    if '*' in path:
        if path.startswith('*'):
            preffix, suffix = '', path[1:]
        else:
            preffix, suffix = path.split('*', maxsplit=1)
        preffix = full_path(preffix)
        if os.path.exists(preffix) is False:
            raise Exception(f"Path {preffix} not found")
        
        choices = {}
        index = 0 
        dir_contents = os.listdir(preffix)
        dir_contents.sort()
        for file in dir_contents:
            if not file.endswith(suffix):
                continue
            index += 1
            choices[f'{index}'] = file

        print("Select the file to process: ")
        for k, v in choices.items():
            print(f'{k}: {v}')

        selected = input("\nEnter the number of the file to process: ")
        if selected not in choices:
            raise Exception(f"Invalid selection {selected}")
        
        path = os.path.join(preffix, choices[selected])
    else:
        path = full_path(path)
    contents = read_worflow_file('', path, set(), {}, '')
    parameters = {
        "workflow": contents,
        "collection": extract_workflow_collections(preffix, contents, {})
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
