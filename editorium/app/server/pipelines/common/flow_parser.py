import random
from typing import List
from uuid import uuid4


class IgnoredItemException(Exception):
    pass


class InvalidItemException(Exception):
    pass


CONFIG_VALIDATOR = lambda task_type, config: True

def convert_value(value: str):
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        if value.lower() in ["yes", "true", "1"]:
            return True
        elif value.lower() in ["no", "false", "0"]:
            return False
    except ValueError:
        pass
    if ',' in value:
        try:
            return [int(v.strip()) for v in value.split(',')]
        except ValueError:
            pass
    if ',' in value:
        try:
            return [float(v.strip()) for v in value.split(',')]
        except ValueError:
            pass
    return value
    
class FlowItem:
    name: str = ""
    config: dict = {}
    task_type: str = ""
    input: dict = {}
    flow_lazy: bool = False
    
    def __init__(self, name: str, task_type: str, input: dict, config: dict, flow_lazy: bool) -> None:
        self.name = name
        self.config = config
        self.input = input
        self.task_type = task_type
        self.flow_lazy = flow_lazy
    
    @classmethod
    def from_lines(cls, lines: List[str], globals: dict, flow_lazy: bool):
        config = {}
        name = str(uuid4())
        input = {}
        task_type = ""
        prompt = []
        negative_prompt = []
        prompt_started = False
        negative_started = False
        for line in lines:
            if line.startswith("#ignore"):
                raise IgnoredItemException("Item is ignored")

            if line.strip() == "#prompt":
                if prompt_started:
                    raise InvalidItemException("Prompt already started")
                prompt_started = True
                continue

            if line.strip() == "#negative":
                if not prompt_started:
                    raise InvalidItemException("Prompt not started")
                if negative_started:
                    raise InvalidItemException("Negative prompt already started")
                negative_started = True
                continue
            
            if prompt_started:
                if negative_started:
                    negative_prompt.append(line)
                else:
                    prompt.append(line)
                continue
            
            if line.startswith("#name="):
                name = line.split("#name=")[1]
                continue  

            if line.startswith("#input="):
                input["default"] = line.split("#input=")[1]
                continue

            if line.startswith("#input."):
                key = line.split("#input.")[1].split("=")[0]
                value = line.split("#input.")[1].split("=")[1]
                input[key] = value
                continue
            
            if line.startswith("#task_type="):
                task_type = line.split("#task_type=")[1]
                continue
            
            if line.startswith("#config."):
                key, value = line.split("#config.")[1].split("=")
                key = key.strip()
                value = convert_value(value.strip())
                if type(value) is str and value.startswith("global://"):
                    global_key = value.split("global://")[1]
                    value = globals.get(global_key, '')
                if key == 'seed' and value == -1:
                    value = random.randint(0, 1000000)
                config[key] = value
                continue

            if line.startswith("#"):
                raise InvalidItemException("The line starts with # but it's not a valid command")
        
        if len(prompt) > 0:
            config['prompt'] = '\n'.join(prompt)
            
        if len(negative_prompt) > 0:
            config['negative_prompt'] = '\n'.join(negative_prompt)
        
        if not task_type:
            raise InvalidItemException(f"task not found task_type: {task_type}")

        if not CONFIG_VALIDATOR(task_type, config):
            raise InvalidItemException(f"Invalid config on task name={name} task_type={task_type}")

        return cls(name, task_type, input, config, flow_lazy)


class FlowStore:
    def __init__(self) -> None:
        self.flows = {}
        self.globals = {}
        
    def add_flow(self, flow: FlowItem):
        self.flows[flow.name] = flow
        
    def parse_flow(self, lines: List[str], flow_lazy: bool):
        try:
            flow = FlowItem.from_lines(lines, self.globals, flow_lazy)
            self.add_flow(flow)
        except IgnoredItemException:
            pass
            

    def find_circular_deps(self, flow: FlowItem, names: set) -> bool:
        dependency = ""
        for value in flow.input.values():
            if value.startswith("task://"):
                dependency = value.split("task://")[1]
                if dependency in names:
                    return True
                names.add(dependency)
                if dependency not in self.flows:
                    raise InvalidItemException(f"Dependency not found {dependency}")
                return self.find_circular_deps(self.flows[dependency], names)
        return False
        
    def validate_circular_dependencies(self):
        for flow in self.flows.values():
            names = set()
            if self.find_circular_deps(flow, names):
                raise InvalidItemException("Circular reference detected")
            
    def validate_repeated_names(self):
        names = set()
        for flow in self.flows.values():
            if flow.name in names:
                raise InvalidItemException("Repeated name detected")
            names.add(flow.name)

    def load(self, lines: List[str]) -> None:
        self.flows = {}
        self.globals = {}
        flow_started = False
        flow_lazy = False
        flow_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('#comment'):
                continue
            
            if line.startswith('#global.'):
                if flow_started:
                    raise InvalidItemException("Global definition inside flow")
                key, value = line.split('#global.')[1].split('=')
                key = key.strip()
                value = convert_value(value)
                if key == 'seed' and value == -1:
                    value = random.randint(0, 1000000)
                self.globals[key] = value
                continue

            if line.startswith('#start'):
                flow_started = True
                continue

            if line.startswith('#end'):
                self.parse_flow(flow_lines, flow_lazy)
                flow_started = False
                flow_lazy = False
                flow_lines = []
                continue
            
            if line.strip() == "#lazy":
                if not flow_started:
                    raise InvalidItemException("#Lazy outside a task")
                flow_lazy = True
                continue

            if flow_started:
                flow_lines.append(line)

        self.validate_repeated_names()
        self.validate_circular_dependencies()
        
    def get_task(self, name: str) -> FlowItem:
        return self.flows.get(name, None)
    
    def iterate(self):
        for flow in self.flows.values():
            yield flow
        

def register_validator(func: callable):
    global CONFIG_VALIDATOR
    CONFIG_VALIDATOR = func
        

flow_store = FlowStore()

__all__ = ["flow_store", "FlowItem", "register_validator", "InvalidItemException", "IgnoredItemException"]