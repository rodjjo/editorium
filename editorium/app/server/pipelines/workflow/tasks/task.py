import os
import random
from zoneinfo import ZoneInfo
import time
from datetime import datetime, timedelta
from importlib import import_module
from typing import List
from copy import deepcopy
from pipelines.common.flow_parser import FlowStore, FlowItem, parse_task_value

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
    def __init__(self, task_type: str, description: str, config_schema=None, input_schema=None, is_api=False):
        self.task_type = task_type
        self.description = description
        self.config_schema = config_schema
        self.input_schema = input_schema
        self.is_api = is_api
            
    def validate_config(self, config: dict):
        if self.config_schema:
            try:
                self.config_schema(context={'from_api': self.is_api}).load(config)
            except Exception as e:
                print(str(e))
                return False
        return True
    
    def _process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        if self.config_schema:
            config = self.config_schema(context={'from_api': self.is_api}).load(config)
        if self.input_schema:
            input = self.input_schema(context={'from_api': self.is_api}).load(input)
        result = self.process_task(base_dir, name, input, config)
        if self.input_schema:
            result = self.input_schema(context={'from_api': self.is_api}).dump(result)
        return result
    
    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        return {}
    
    @classmethod
    def register(cls,task_type: str, description: str):
        instance = cls(task_type, description, is_api=False)
        WorkflowTaskManager.add_task(instance)


class WorkflowTaskManager:
    _tasks: dict = {}
    
    def first_existing_task(self, taskname: str):
        if '|' in taskname:
            first, second = taskname.split('|', maxsplit=1)
            first = first.strip()
            second = second.strip()
            if first in self.results:
                return first
            print(f"First task {first} not found, trying second task")
            return second
        return taskname
    
    @property
    def tasks(self) -> dict:
        return WorkflowTaskManager._tasks
    
    def __init__(self, workflow_collection: dict):
        self.flow_store = FlowStore(self)
        self.workflow_collection = workflow_collection
        self.results = {}
    
    @classmethod
    def add_task(cls, task: WorkflowTask):
        if task.task_type in cls._tasks:
            raise ValueError(f'Task {task.task_type} already exists')
        cls._tasks[task.task_type] = task
        
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
    
    def process_task(self, base_dir, item: FlowItem, task_stack: set):
        if item.name in self.results:
            if self.results[item.name].get('_injected'):
                return self.results[item.name]

        if item.task_type not in self.tasks:
            raise ValueError(f'Task {item.task_type} is not registered')

        if item.name in self.results:
            return self.accept_resolved_value(self.results[item.name]['_item'], self.results[item.name])
        
        if item.name in task_stack:
            raise ValueError(f'Circular dependency detected in task {item.name}')
      
        task_stack.add(item.name)

        resolved_inputs = {}
        for input, value in item.input.items():
            value = value.strip()
            if value:
                if value.startswith('task://'):
                    task_name = self.first_existing_task(value.split('task://')[1])
                else:
                    task_name = self.first_existing_task(value)
                if task_name in self.results:
                    print(f'Using cached result of {task_name} for task {item.name}')
                    resolved = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name])
                else:
                    print(f'Processing task {task_name} to resolve input for task {item.name}')
                    self.process_task(base_dir, self.flow_store.get_task(task_name), task_stack)
                    resolved = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name])
            else:
                print(f'Empty input for task {item.name}')
                resolved = {}
            resolved_inputs[input] = resolved

        if item.config.get('prompt', '').startswith('task://'):
            prompt = item.config.get('prompt', '')
            task_name = self.first_existing_task(prompt.split('task://')[1])
            if task_name not in self.results:
                self.process_task(base_dir, self.flow_store.get_task(task_name), task_stack)
            item.config['prompt'] = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', ''))

        if item.config.get('negative_prompt', '').startswith('task://'):
            negative_prompt = item.config.get('negative_prompt', '')
            task_name = self.first_existing_task(negative_prompt.split('task://')[1])
            if task_name not in self.results:
                self.process_task(base_dir, self.flow_store.get_task(task_name), task_stack)
            item.config['negative_prompt'] = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', ''))
            
        for key, value in item.config.items():
            if key in ['prompt', 'negative_prompt', 'globals']:
                while '<insert_task:' in value:
                    start = value.find('<insert_task:')
                    end = value.find('>', start)
                    if end == -1:
                        raise ValueError(f'Invalid insert_task tag in task {item.name}')
                    task_name = value[start + 13:end]
                    
                    if task_name not in self.results:
                        self.process_task(base_dir, self.flow_store.get_task(task_name), task_stack)
                    task_value = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', '') or self.results[task_name].get('result', ''))
                    if type(task_value) is list:
                        task_value = task_value[0]
                    if type(task_value) is not str:
                        raise ValueError(f'Task {task_name} did not return a string')
                    value = value[:start] + task_value + value[end + 1:]
                item.config[key] = value
                continue

            if type(value) is str and value.startswith('task://'):
                task_name = value.split('task://')[1]
                default_value = None
                if ':' in task_name:
                    task_name, default_value = task_name.split(':', maxsplit=1)
                    task_name = self.first_existing_task(task_name)
                    if task_name in self.flow_store.flows:
                        default_value = None
                else:
                    task_name = self.first_existing_task(task_name)
                if task_name not in self.results:
                    if default_value is None:
                        self.process_task(base_dir, self.flow_store.get_task(task_name), task_stack)
                if default_value is None:
                    task_value = self.accept_resolved_value(self.results[task_name]['_item'], self.results[task_name].get('default', '') or self.results[task_name].get('result', ''))
                else:
                    task_value = default_value
                if type(task_value) is list:
                    task_value = task_value[0]
                if type(task_value) is not str:
                    raise ValueError(f'Task {task_name} did not return a string')
                item.config[key] = parse_task_value(task_value)

        try:
            def execute_manager(path, inject_result, result_task_name, parent_dir):
                global_seed = self.flow_store.globals.get('seed', -1)
                new_manager = WorkflowTaskManager(self.workflow_collection)
                for k, v in inject_result.items():
                    if type(v) is not dict:
                        continue
                    print(f'Injecting result of {k} into the results')
                    v = {
                        '_injected': True,
                        **v,
                    }
                    new_manager.results[k] = v
                    
                new_manager.execute(self.workflow_collection[path], use_global_seed=global_seed, disable_repeat=True, parent_dir=parent_dir, clear_result=False)
                if result_task_name in new_manager.results:
                    return new_manager.results[result_task_name]
                else:
                    raise ValueError(f'Task {result_task_name} not found in {path}')
            
            task_result = self.tasks[item.task_type]._process_task(
                base_dir,
                item.name,
                deepcopy(resolved_inputs), 
                {
                    **deepcopy(item.config),
                    'globals': {
                        **deepcopy(self.flow_store.globals),
                        'execute_manager': execute_manager, # contents: List[str], use_global_seed=None, disable_repeat=False, 
                    },
                }
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
                    self.process_task(base_dir, self.flow_store.get_task(value), task_stack)
                else:
                    print(f'Empty task name in decision task {item.name}')

    def save_workflow(self, contents: List[str], filepath: str):
        with open(filepath, 'w') as f:
            for line in contents:
                f.write(line + '\n')

        
    def execute(self, contents: List[str], use_global_seed=None, disable_repeat=False, parent_dir='', clear_result=True) -> dict:
        should_repeat = False
        found_global_seed = False
        found_global_debug = False
        for index, line in enumerate(contents):
            if disable_repeat is False and line.startswith('#global.repeat='):
                should_repeat = line.split('#global.repeat=')[1].lower() in ['true', '1', 'yes', 'on', 'sure']
                break

            if line.startswith('#global.seed='):
                if use_global_seed is not None:
                    contents[index] = f'#global.seed={use_global_seed}'
                found_global_seed = True

            if line.startswith('#global.debug='):
                found_global_debug = line.split('#global.debug=')[1].lower() in ['true', '1', 'yes', 'on', 'sure']

        if not found_global_seed:
            contents = [f'#global.seed={use_global_seed or random.randint(0, 1000000)}'] + contents

        task_run_count = 0
        while True:
            try:
                if clear_result:
                    self.results = {}
                clear_result = True
                # dirname = current date and time in format YYYYMMDD-HH-MM-SS.00 in local time
                time.sleep(0.5)
                dirname = now_on_tz().strftime('%Y%m%d-%H-%M-%S.%f')[:-2]
                if parent_dir:
                    dirpath = os.path.join(BASE_DIR, "workflow-outputs", parent_dir, dirname)
                else:
                    dirpath = os.path.join(BASE_DIR, "workflow-outputs", dirname)
                os.makedirs(dirpath, exist_ok=True)
                
                self.flow_store.load(contents)
                
                for item in self.flow_store.iterate():
                    if item.flow_lazy:
                        continue
                    self.process_task(dirpath, item, set())
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


def register_workflow_tasks():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.startswith('task_') and file.endswith('.py'):
            module = import_module(f'pipelines.workflow.tasks.{file[:-3]}')
            if hasattr(module, 'register'):
                module.register()
            else:
                print(f'Warning: {file} does not have a register function')

register_workflow_tasks()


def get_workflow_manager(workflow_collection: dict):
    return WorkflowTaskManager(workflow_collection)


__all__ = ['get_workflow_manager', 'WorkflowTask', 'WorkflowTaskManager']
