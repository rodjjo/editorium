from typing import List
from uuid import uuid4


class IgnoredItemException(Exception):
    pass


class InvalidItemException(Exception):
    pass


CONFIG_VALIDATOR = lambda task_type, config: True


class FlowItem:
    name: str = ""
    config: dict = {}
    task_type: str = ""
    input: dict = {}
    
    def __init__(self, name: str, task_type: str, input: dict, config: dict) -> None:
        self.name = name
        self.config = config
        self.input = input
        self.task_type = task_type
    
    @classmethod
    def from_lines(cls, lines: List[str]):
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
                value = value.strip()
                converted = False
                try:
                    value = float(value)
                    converted = True
                except ValueError:
                    pass
                if not converted:
                    try:
                        value = int(value)
                        converted = True
                    except ValueError:
                        pass
                if not converted:
                    try:
                        converted = True
                        if value.lower() in ["yes", "true", "1"]:
                            value = True
                        elif value.lower() in ["no", "false", "0"]:
                            value = False
                        else:
                            converted = False
                    except ValueError:
                        pass
                if not converted:
                    if ',' in value:
                        try:
                            value = [float(v.strip()) for v in value.split(',')]
                            converted = True
                        except ValueError:
                            pass
                if not converted:
                    if ',' in value:
                        try:
                            value = [int(v.strip()) for v in value.split(',')]
                            converted = True
                        except ValueError:
                            pass
                config[key] = value
        
        if len(prompt) > 0:
            config['prompt'] = '\n'.join(prompt)
            
        if len(negative_prompt) > 0:
            config['negative_prompt'] = '\n'.join(negative_prompt)
        
        if not config:
            raise InvalidItemException("No config found")

        if not task_type:
            raise InvalidItemException("No task type found")

        if not CONFIG_VALIDATOR(task_type, config):
            raise InvalidItemException("Invalid config")

        return cls(name, task_type, input, config)


class FlowStore:
    def __init__(self) -> None:
        self.flows = {}
        
    def add_flow(self, flow: FlowItem):
        self.flows[flow.name] = flow
        
    def parse_flow(self, lines: List[str]):
        try:
            flow = FlowItem.from_lines(lines)
            print(f"Adding flow task type: {flow.task_type}")
            self.add_flow(flow)
        except IgnoredItemException:
            print("Item ignored")
            

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
        flow_started = False
        flow_lines = []
        print(f"Loading flows line count: {len(lines)}")
        
        for line in lines:
            line = line.strip()
            if line.startswith('#comment'):
                continue

            if line.startswith('#start'):
                flow_started = True
                continue

            if line.startswith('#end'):
                flow_started = False
                print("Parsing flow task")
                self.parse_flow(flow_lines)
                flow_lines = []
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