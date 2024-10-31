from .task import WorkflowTask
from marshmallow import Schema, fields


class ReturnInputTaskSchema(Schema):
    name = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class ReturnInputTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = ReturnInputTaskSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing select input task")
        params = ReturnInputTaskSchema().load(config)
        name = params['name']
        input = input.get("default", {})
        if name not in input:
            raise ValueError(f"Input {name} not found in the task return inputs")
        return input[name]


def register():
    ReturnInputTask.register("select-input", "Select an input from the task return-inputs")
