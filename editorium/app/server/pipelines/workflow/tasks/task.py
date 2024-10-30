import os
import random
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from importlib import import_module
from typing import List
from copy import deepcopy
from pipelines.common.flow_parser import register_validator, flow_store, FlowItem, parse_task_value

BASE_DIR = '/app/output_dir'


def now_on_tz():
    tz = os.environ.get('TZ', 'UTC')
    try:
        hours = int(tz)
        if hours < 0:
            hours = -hours
            return datetime.now() - timedelta(hours=hours)
        return datetime.now() + timedelta(hours=hours)
    except ValueError:
        tz_info = ZoneInfo(tz)
        return datetime.now().astimezone(tz_info)


class WorkflowTask:
    def __init__(self, task_type: str, description: str):
        self.task_type = task_type
        self.description = description
        
    def validate_config(self, config: dict):
        return True
    
    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        return {}
    
    @classmethod
    def register(cls,task_type: str, description: str):
        instance = cls(task_type, description)
        task_manager.add_task(instance)


class WorkflowTaskManager:
    tasks: dict = {}
    results = {}

    def __init__(self):
        pass
    
    def add_task(self, task: WorkflowTask):
        if task.task_type in self.tasks:
            raise ValueError(f'Task {task.task_type} already exists')
        self.tasks[task.task_type] = task
        
    def validate_config(self, task_type: str, config: dict):
        if task_type not in self.tasks:
            raise ValueError(f'Task {task_type} not found')
        return self.tasks[task_type].validate_config(config)
    
    def accept_resolved_value(self, item, value):
        if item.decision:
            item_result = self.results[item.name]
            if type(item_result) is not dict:
                raise ValueError(f'Decision Task {item.name} did not return a dictionary')
            if 'default' not in item_result:
                raise ValueError(f'Decision Task {item.name} did not return a default value')
            default = item_result['default']
            if type(default) is not list:
                raise ValueError(f'Decision Task {item.name} default value is not a list')
            if len(default) == 0:
                raise ValueError(f'Decision Task {item.name} default value is empty')
            result_task_name = default[0]
            if result_task_name not in self.results:
                raise ValueError(f'Decision Task {item.name} default value {result_task_name} not found')
            return self.accept_resolved_value(self.results[result_task_name]['_item'], self.results[result_task_name])
        return value
    
    def process_task(self, base_dir, item: FlowItem, callback: callable, task_stack: set):
        if item.task_type not in self.tasks:
            raise ValueError(f'Task {item.task_type} is not registered')

        if item.name in self.results:
            return self.results[item.name]
        
        if item.name in task_stack:
            raise ValueError(f'Circular dependency detected in task {item.name}')
      
        task_stack.add(item.name)

        resolved_inputs = {}
        for input, value in item.input.items():
            if value.startswith('task://'):
                task_name = value.split('task://')[1]
                if task_name in self.results:
                    print(f'Using cached result of {task_name} for task {item.name}')
                    resolved = self.results[task_name]
                else:
                    print(f'Processing task {task_name} to resolve input for task {item.name}')
                    self.process_task(base_dir, flow_store.get_task(task_name), callback, task_stack)
                    resolved = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name])
            elif value:
                print(f'Using literal value {value} for task {item.name}')
                resolved = {
                    "result": [value]
                }
            else:
                print(f'Empty input for task {item.name}')
                resolved = {}
            resolved_inputs[input] = resolved

        if item.config.get('prompt', '').startswith('from://'):
            prompt = item.config.get('prompt', '')
            task_name = prompt.split('from://')[1]
            if task_name not in self.results:
                self.process_task(base_dir, flow_store.get_task(task_name), callback, task_stack)
            item.config['prompt'] = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', ''))

        if item.config.get('negative_prompt', '').startswith('from://'):
            negative_prompt = item.config.get('negative_prompt', '')
            task_name = negative_prompt.split('from://')[1]
            if task_name not in self.results:
                self.process_task(base_dir, flow_store.get_task(task_name), callback, task_stack)
            item.config['negative_prompt'] = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', ''))
            
        for key, value in item.config.items():
            if key in ['prompt', 'negative_prompt', 'globals']:
                while '<insert_from:' in value:
                    start = value.find('<insert_from:')
                    end = value.find('>', start)
                    if end == -1:
                        raise ValueError(f'Invalid insert_from tag in task {item.name}')
                    task_name = value[start + 13:end]
                    if task_name not in self.results:
                        self.process_task(base_dir, flow_store.get_task(task_name), callback, task_stack)
                    task_value = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', '') or self.results[task_name].get('result', ''))
                    if type(task_value) is list:
                        task_value = task_value[0]
                    if type(task_value) is not str:
                        raise ValueError(f'Task {task_name} did not return a string')
                    value = value[:start] + task_value + value[end + 1:]
                item.config[key] = value
                continue

            if type(value) is str and value.startswith('from://'):
                task_name = value.split('from://')[1]
                if task_name not in self.results:
                    self.process_task(base_dir, flow_store.get_task(task_name), callback, task_stack)
                task_value = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', '') or self.results[task_name].get('result', ''))
                if type(task_value) is list:
                    task_value = task_value[0]
                if type(task_value) is not str:
                    raise ValueError(f'Task {task_name} did not return a string')
                item.config[key] = parse_task_value(task_value)

        try:
            task_result = self.tasks[item.task_type].process_task(
                base_dir,
                item.name,
                deepcopy(resolved_inputs), 
                {
                    **deepcopy(item.config),
                    'globals': deepcopy(flow_store.globals),
                },
                callback
            )
        except Exception as e:
            print(f"Error processing task {item.name}: {str(e)}")
            raise
            
        task_result['_item'] = item
        self.results[item.name] = task_result

        if item.decision:
            if type(task_result) is not dict:
                raise ValueError(f'Decision Task {item.name} did not return a dictionary')
            if 'default' not in task_result:
                raise ValueError(f'Decision Task {item.name} did not return a default value')
            default = task_result['default']
            if type(default) is not list:
                raise ValueError(f'Decision Task {item.name} default value is not a list')
            for i, value in enumerate(default):
                if type(value) is not str:
                    raise ValueError(f'Decision Task {item.name} default value at index {i} is not a string')
            for value in default:
                value = value.strip()
                if value != '':
                    if value == item.name:
                        raise ValueError(f'Decision Task {item.name} cannot reference itself')
                    self.process_task(base_dir, flow_store.get_task(value), callback, task_stack)
                else:
                    print(f'Empty task name in decision task {item.name}')

    def save_workflow(self, contents: List[str], filepath: str):
        with open(filepath, 'w') as f:
            for line in contents:
                f.write(line + '\n')

        
    def execute(self, contents: List[str], callback = None) -> dict:
        should_repeat = True
        found_global_seed = False
        found_global_debug = False
        for line in contents:
            if line.startswith('#global.repeat='):
                should_repeat = line.split('#global.repeat=')[1].lower() in ['true', '1', 'yes', 'on', 'sure']
                break
            if line.startswith('#global.seed='):
                found_global_seed = True
            if line.startswith('#global.debug='):
                found_global_debug = line.split('#global.debug=')[1].lower() in ['true', '1', 'yes', 'on', 'sure']
        if not found_global_seed:
            contents = [f'#global.seed={random.randint(0, 1000000)}'] + contents

        task_run_count = 0
        while True:
            try:
                self.results = {}
                # dirname = current date and time in format YYYYMMDD-HH-MM-SS in local time
                dirname = now_on_tz().strftime('%Y%m%d-%H-%M-%S')
                dirpath = os.path.join(BASE_DIR, "workflow-outputs", dirname)
                os.makedirs(dirpath, exist_ok=True)
                
                flow_store.load(contents)
                
                for item in flow_store.iterate():
                    if item.flow_lazy:
                        continue
                    self.process_task(dirpath, item, callback, set())
                    task_run_count += 1
                if not should_repeat:
                    break

                for i, l in enumerate(contents):
                    if l.startswith('#config.seed=') and not l.startswith('#config.seed=global://seed'):
                        contents[i] = f'#config.seed={random.randint(0, 1000000)}'
                    elif l.startswith('#global.seed='):
                        contents[i] = f'#global.seed={random.randint(0, 1000000)}'

                if task_run_count == 0:
                    raise ValueError("No tasks were executed")

                if found_global_debug:
                    self.save_workflow(contents, os.path.join(dirpath, 'workflow.txt'))
            except:
                self.save_workflow(contents, os.path.join(dirpath, 'workflow.txt'))
                raise 
        return { 
            "success": True 
        }
        
    def get_registered_tasks(self):
        registered_tasks = []
        for task in self.tasks.values():
            registered_tasks.append({
                "task_type": task.task_type,
                "description": task.description
            })
        return registered_tasks


task_manager = WorkflowTaskManager()


def validate_config(task_type: str, config: dict):
    return task_manager.validate_config(task_type, config)


def register_workflow_tasks():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.startswith('task_') and file.endswith('.py'):
            module = import_module(f'pipelines.workflow.tasks.{file[:-3]}')
            if hasattr(module, 'register'):
                module.register()
            else:
                print(f'Warning: {file} does not have a register function')


def get_workflow_manager():
    return task_manager


register_workflow_tasks()
register_validator(validate_config)


__all__ = ['get_workflow_manager', 'WorkflowTask']
