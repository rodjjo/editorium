from .task import WorkflowTask
from marshmallow import Schema, fields


class SelectInputTaskSchema(Schema):
    name = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class SelectInputTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=SelectInputTaskSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing select input task")
        name = config['name']
        input = input.get("default", {}).get("data", {})
        if name not in input:
            raise ValueError(f"Input {name} not found in the task return inputs")
        return input[name]


def register():
    SelectInputTask.register("select-input", "Select an input from the task return-inputs")
